---
title: "CNN 이미지 분류 실습(4) — 학습을 몇 배 빠르게? DataLoader·AMP·Gradient Accumulation·torch.compile로 훈련 최적화"
date: 2026-09-11 14:45:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, cnn, resnet, amp, mixed-precision, torch-compile, gradient-accumulation, training-optimization, deep-learning, performance]
description: "ResNet-18을 CIFAR-10으로 학습할 때 처리량(img/sec)을 기준으로, num_workers·pin_memory → AMP(혼합 정밀도) → Gradient Accumulation → torch.compile을 단계별로 쌓아가며 얼마나 빨라지는지 직접 측정한다. 체크포인트·스케줄러·조기종료 같은 훈련 관리 기법까지 정리했다."
---

> [지난 편]({% post_url 2026-09-11-cnn-classification-3-cnn-scratch-vs-transfer-learning %})까지는 "얼마나 정확한 모델을 만드는가"에 집중했다면, 이번 편은 **정확도가 아니라 "같은 모델·같은 데이터를 얼마나 빠르고 가볍게 학습시킬 수 있는가"**를 다룹니다. DataLoader 최적화 → AMP(혼합 정밀도) → Gradient Accumulation → torch.compile을 하나씩 쌓아가며 처리량(img/sec)을 직접 측정하고, 체크포인트·스케줄러·조기종료 같은 훈련 관리 기법도 함께 실습합니다.

## 전체 그림

```
STEP 0: 데이터·모델·측정기 준비
STEP 1: 기준선(baseline) 측정
STEP 2: DataLoader 최적화 (num_workers, pin_memory)
STEP 3: AMP (혼합 정밀도)
STEP 4: Gradient Accumulation
STEP 5: 훈련 관리 (재현성·체크포인트·스케줄러·조기종료)
STEP 6: torch.compile
미니 프로젝트: 전체 결과 비교
```

![DataLoader 최적화, AMP, Gradient Accumulation, torch.compile을 순서대로 쌓아가며 ResNet-18 학습 처리량(img/sec)이 기준선 대비 몇 배 빨라지는지 보여주는 최적화 파이프라인 다이어그램](/assets/img/posts/optimization/diagram4-optimization-pipeline.svg)

---

## 준비 — 환경 설정

```python
import subprocess, os, shutil, time, copy
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)  # 그래프 한글 폰트
shutil.rmtree(os.path.expanduser('~/.cache/matplotlib'), ignore_errors=True)

import numpy as np, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
_fp = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(_fp):
    fm.fontManager.addfont(_fp); plt.rcParams['font.family'] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True; plt.rcParams['grid.alpha'] = 0.3
TEAL = '#0D9488'; GRAY = '#94A3B8'; AMBER = '#F59E0B'; RED = '#B91C1C'

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvm
from torchvision import transforms
torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IS_CUDA = (device.type == 'cuda')   # GPU 유무를 미리 저장 — 이후 모든 최적화 분기에 사용
print('준비 완료 · device:', device)
if IS_CUDA:
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('※ CPU 모드 — 코드는 돌지만 속도/메모리 숫자는 참고용입니다. Colab이면 런타임을 GPU로 바꾸세요.')
```

`IS_CUDA` 변수로 GPU 유무를 미리 확인해두고, 앞으로 나올 모든 최적화 기법(AMP, pin_memory 등)이 GPU에서만 의미 있는 기능이라 이 변수로 계속 분기 처리합니다.

## STEP 0 — 데이터·모델·측정기 준비

측정 대상은 **ResNet-18을 CIFAR-10으로 학습**할 때의 처리량과 메모리입니다.

```python
# 실습 설정 (여기 숫자만 바꾸면 전체가 따라 바뀝니다)
IMG_SIZE = 128     # 입력 크기 (클수록 연산↑ · AMP 효과 뚜렷). 빠르게: 96
N_TRAIN  = 5000    # 학습 이미지 수
N_TEST   = 1000    # 테스트 이미지 수
BATCH    = 64      # 기본 배치
STEPS    = 25      # 측정 스텝 수
WARMUP   = 5       # 워밍업 스텝 수
MODEL    = 'resnet18'   # 'resnet18' / 'resnet34' / 'resnet50' 로 바꿔 크기별 속도·메모리 비교 가능

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# ── 전체 대신 필요한 만큼만 스트리밍 ──
!pip install -q datasets
from datasets import load_dataset
_tr = load_dataset('uoft-cs/cifar10', split='train', streaming=True)
_te = load_dataset('uoft-cs/cifar10', split='test',  streaming=True)
train_raw = list(_tr.take(N_TRAIN))
test_raw  = list(_te.take(N_TEST))

class DS(Dataset):
    # raw: [{'img': PIL.Image, 'label': int}, ...] 리스트를 감싸는 래퍼.
    # __getitem__은 (전처리된 이미지 텐서, 정수 라벨) 튜플을 돌려줌 — DataLoader가 요구하는 규약.
    def __init__(s, raw, tf): s.raw, s.tf = raw, tf
    def __len__(s): return len(s.raw)
    def __getitem__(s, i):
        e = s.raw[i]
        return s.tf(e['img'].convert('RGB')), int(e['label'])

train_ds, test_ds = DS(train_raw, tf), DS(test_raw, tf)

crit = nn.CrossEntropyLoss()

def make_model():
    m = getattr(tvm, MODEL)(weights=None)      # 사전학습 X — 순수 아키텍처 속도 측정
    m.fc = nn.Linear(m.fc.in_features, 10)      # CIFAR-10: 10클래스
    return m.to(device)

print(f'데이터 준비 완료 — 학습 {len(train_ds)}장 / 테스트 {len(test_ds)}장, {IMG_SIZE}px, 배치 {BATCH}, 모델 {MODEL}')
```

> `make_model()`에서 `weights=None`인 이유: 이 실습의 목적은 "얼마나 정확한가"가 아니라 "구조 자체가 얼마나 빨리 도는가"이므로, 가중치 유무와 상관없이 순수 연산 속도만 측정합니다. `getattr(tvm, MODEL)`은 `MODEL` 문자열 값(`'resnet18'` 등)에 해당하는 함수를 동적으로 꺼내와서, `MODEL` 값 하나만 바꾸면 다른 모델도 코드 수정 없이 테스트할 수 있게 해줍니다.

### 측정기(벤치마크 함수) 만들기

```python
# 처리량(img/sec) + 최대 메모리(GB) 측정기 — 데이터가 어디서든(무한 반복) 안 깨지게
SCORE, MEM = {}, {}   # SCORE/MEM: {실험 이름(str): 측정값(float)} 형태의 점수판. STEP마다 여기에 결과를 채워감.

def run_bench(loader, use_amp=False, steps=STEPS, warmup=WARMUP):
    # 반환값: (thru, mem) = (초당 처리 이미지 수, 이번 측정 구간의 GPU 최대 메모리 사용량 GB)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    amp_on = use_amp and IS_CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    model.train()
    if IS_CUDA: torch.cuda.reset_peak_memory_stats()   # 이번 측정 구간만의 메모리를 재기 위해 리셋

    it = iter(loader)
    def nxt():
        nonlocal it   # 바깥 함수의 it 변수를 그대로 재사용하겠다는 선언
        try: return next(it)
        except StopIteration:
            it = iter(loader); return next(it)   # 데이터가 떨어지면 처음부터 다시 시작 (무한 반복)

    def one_step():
        x, y = nxt(); x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device.type, enabled=amp_on):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        return x.size(0)

    for _ in range(warmup): one_step()          # 워밍업(첫 스텝의 느린 초기화 시간 제외)
    if IS_CUDA: torch.cuda.synchronize()         # GPU는 비동기라 시간 재기 전 동기화 필수
    t0 = time.time(); n = 0
    for _ in range(steps): n += one_step()
    if IS_CUDA: torch.cuda.synchronize()
    dt = time.time() - t0
    thru = n / dt
    mem = torch.cuda.max_memory_allocated() / 1e9 if IS_CUDA else 0.0
    return thru, mem

print('측정기 준비 완료 — run_bench(loader, use_amp=False) 로 호출')
```

**핵심 포인트 3가지**
- **워밍업(warmup)**: GPU는 처음 몇 번의 연산에서 CUDA 커널을 초기화하는 시간이 걸려서, 이 구간을 측정에서 빼야 실제 반복 학습 속도를 정확히 잴 수 있습니다.
- **`torch.cuda.synchronize()`**: GPU 연산은 비동기(asynchronous)입니다. CPU가 GPU에 명령만 던지고 결과를 기다리지 않고 바로 다음 코드로 넘어가버리기 때문에, 동기화 없이 시간을 재면 GPU가 아직 계산 중인데 "다 끝났다"고 착각해서 실제보다 빠르게 측정됩니다. `synchronize()`로 GPU가 진짜 끝날 때까지 기다린 후 시간을 재야 정확합니다.
- **`nxt()`의 `nonlocal it`**: 내부 함수(`nxt`)가 바깥 함수(`run_bench`)의 변수 `it`을 새로 만드는 게 아니라 그대로 재사용하겠다고 선언하는 키워드입니다. 이 함수는 DataLoader의 배치가 다 떨어지면(`StopIteration`) 자동으로 처음부터 다시 시작해서, 정해진 `steps`만큼 무한히 반복할 수 있게 해줍니다.

## STEP 1 — 기준선(baseline) 측정

```python
base_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=0)   # 최적화 전
tp_base, mem_base = run_bench(base_loader, use_amp=False)
SCORE['① 기준선\n(workers0·FP32)'] = tp_base
MEM['① 기준선'] = mem_base
msg = f'🏁 기준선: {tp_base:,.0f} img/sec'
if IS_CUDA: msg += f'   |   peak memory {mem_base:.2f} GB'
print(msg)
```

아무 최적화도 안 한 상태(`num_workers=0`, FP32)를 기준점으로 잽니다. 이후 모든 개선이 "이 숫자보다 몇 배 빨라졌는가"로 비교됩니다.

## STEP 2 — DataLoader 최적화 (num_workers · pin_memory)

```python
cands = [0, 2, 4]
best_nw, best_tp, best_mem = 0, 0, mem_base
for nw in cands:
    ld = DataLoader(train_ds, batch_size=BATCH, num_workers=nw, pin_memory=IS_CUDA)
    tp, mem = run_bench(ld, use_amp=False)
    print(f'num_workers={nw}: {tp:,.0f} img/sec')
    if tp > best_tp: best_tp, best_nw, best_mem = tp, nw, mem
print(f'가장 빠른 num_workers = {best_nw}')
SCORE['② +DataLoader\n(workers·pin)'] = best_tp
MEM['② +DataLoader'] = best_mem   # 가장 빠른 num_workers 설정에서 실제로 측정된 메모리값
```

- **`num_workers`**: 데이터를 불러오고 전처리하는 작업을 몇 개의 별도 프로세스가 병렬로 미리 준비해둘지 정합니다. 0이면 GPU가 계산하는 동안 데이터 준비가 멈춰있지만(직렬), 2~4면 백그라운드에서 다음 배치를 미리 준비해둬서 GPU가 놀지 않게 만듭니다.
- **`pin_memory=IS_CUDA`**: CPU 메모리를 "고정(pinned)"해두면 CPU→GPU 전송 속도가 빨라집니다. GPU를 쓸 때만 의미가 있어서 `IS_CUDA` 조건으로 켜고 끕니다.
- 여러 `num_workers` 값을 실제로 돌려보고 가장 빠른 값을 자동으로 선택해서 `best_nw`에 저장해두고, 이후 STEP들에서 계속 재사용합니다.
- `run_bench()`는 `(처리량, 메모리)`를 함께 반환하므로, 최고 처리량을 낸 설정의 메모리값(`best_mem`)도 같이 추적해서 `MEM`에 기록합니다. STEP 1의 기준선 메모리를 재사용하지 않고 매번 새로 측정한 값을 쓰는 게 핵심입니다.

## STEP 3 — AMP (혼합 정밀도, Mixed Precision)

```python
opt_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=best_nw, pin_memory=IS_CUDA)
tp_amp, mem_amp = run_bench(opt_loader, use_amp=True)     # ← AMP 켬
SCORE['③ +AMP\n(fp16)'] = tp_amp
MEM['③ +AMP'] = mem_amp
msg = f'AMP: {tp_amp:,.0f} img/sec  (기준선 대비 {tp_amp/tp_base:.2f}배)'
if IS_CUDA: msg += f'   |   memory {mem_amp:.2f} GB (기준선 {mem_base:.2f} GB)'
print(msg)
if not IS_CUDA: print('※ CPU에선 AMP 효과가 없어 숫자 차이가 안 납니다. GPU에서 확인하세요.')
```

연산을 보통의 32비트(FP32) 대신 **16비트(FP16)**로 처리해 더 빠르고 메모리도 적게 씁니다. 최신 GPU는 16비트 연산 전용 가속 하드웨어(Tensor Core)가 있어서 속도 향상 폭이 큽니다.

- **`torch.autocast`**: 이 블록 안의 연산을 자동으로 FP16으로 계산하게 해주는 컨텍스트
- **`GradScaler`**: FP16은 표현 범위가 좁아서 아주 작은 기울기(gradient)가 0으로 사라지는 문제(underflow)가 생길 수 있습니다. `GradScaler`가 손실(loss) 값을 미리 크게 곱해(scale) 기울기가 사라지지 않게 키운 뒤, 실제 가중치 업데이트 직전(`scaler.step()`)에 다시 원래 크기로 되돌려줍니다.

## STEP 4 — Gradient Accumulation

메모리가 부족해 큰 배치를 못 쓸 때, **여러 미니배치의 기울기를 모아 한 번에 업데이트**하면 큰 배치 효과를 낼 수 있습니다.

```python
def train_accum(accum_steps=4, micro_batch=16, n_updates=10):
    model = make_model(); opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ld = DataLoader(train_ds, batch_size=micro_batch, num_workers=best_nw, shuffle=True)
    it = iter(ld); model.train(); opt.zero_grad()
    updates = 0; step = 0
    while updates < n_updates:
        try: x, y = next(it)
        except StopIteration: it = iter(ld); x, y = next(it)
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y) / accum_steps      # ← 누적 스텝 수로 나눔
        loss.backward()
        step += 1
        if step % accum_steps == 0:                 # accum_steps번마다 한 번 업데이트
            opt.step(); opt.zero_grad(); updates += 1
    print(f'유효 배치 = micro_batch({micro_batch}) × accum_steps({accum_steps}) = {micro_batch*accum_steps}')
    print(f'물리 배치는 {micro_batch}장이라 메모리는 작게 쓰면서, 배치 {micro_batch*accum_steps} 효과를 냄')

train_accum(accum_steps=4, micro_batch=16)
```

물리적으로는 `micro_batch=16`장씩만 메모리에 올리지만, `accum_steps=4`번 기울기를 모아서 한 번에 업데이트하므로 효과는 16×4=64장 배치로 학습한 것과 같아집니다. `loss / accum_steps`로 미리 나누는 이유는, 여러 번 `.backward()`를 하면 기울기가 그 횟수만큼 더해지는데, 미리 나눠두면 합쳐진 기울기가 큰 배치 한 번의 **평균 기울기**와 수학적으로 같아지기 때문입니다.

## STEP 5 — 훈련 관리 (재현성 · 체크포인트 · 스케줄러 · 조기종료)

### 재현성

```python
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if IS_CUDA: torch.cuda.manual_seed_all(seed)

set_seed(42)
print('시드 고정 완료')
```

파이썬 기본 random, numpy, torch(CPU), torch(GPU) **네 군데 모두** 시드를 고정해야 완전히 같은 결과가 재현됩니다.

### 체크포인트 저장/복원

```python
model = make_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)   # 학습률 스케줄러

def save_ckpt(path, epoch):
    # 저장되는 딕셔너리 구조: {'model': state_dict, 'opt': state_dict, 'sched': state_dict, 'epoch': int}
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'sched': sched.state_dict(), 'epoch': epoch}, path)

def load_ckpt(path):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched'])
    return ck['epoch']

save_ckpt('ckpt.pt', epoch=3)
print('저장 완료 → 복원한 epoch:', load_ckpt('ckpt.pt'))
print('현재 학습률:', opt.param_groups[0]['lr'])
```

> **왜 모델 가중치만이 아니라 옵티마이저·스케줄러·에폭까지 저장해야 할까요?** 학습을 이어서 하려면 모델 가중치뿐 아니라 옵티마이저의 내부 상태(모멘텀 등)와 스케줄러가 지금 몇 번째 단계인지, 몇 번째 에폭이었는지까지 저장해야 정확히 같은 지점에서 재개됩니다. 가중치만 저장하면 옵티마이저가 초기화되어 학습이 흔들립니다.

> **주의**: 체크포인트 저장/불러오기 자체는 "수동 기능"입니다. "중단된 곳부터 자동으로 이어서 시작"하려면, 학습 시작 시 파일 존재 여부를 확인해서 있으면 불러오고, 학습 중 주기적으로 저장하는 로직을 **직접 학습 루프에 넣어줘야** 합니다. 예:
> ```python
> start_epoch = 0
> if os.path.exists('ckpt.pt'):
>     start_epoch = load_ckpt('ckpt.pt')
> for epoch in range(start_epoch, total_epochs):
>     train_one_epoch(...)
>     save_ckpt('ckpt.pt', epoch)   # 매 에폭 끝날 때마다 저장해 다음 중단에 대비
> ```

### 조기 종료 (Early Stopping)

```python
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience; self.best = float('inf'); self.wait = 0; self.stop = False

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss; self.wait = 0       # 개선됨 → 대기 초기화
        else:
            self.wait += 1
            if self.wait >= self.patience: self.stop = True
        return self.stop

# 데모: 검증 손실이 오르기 시작하면 멈춤
es = EarlyStopping(patience=2)
for ep, vloss in enumerate([1.0, 0.8, 0.7, 0.72, 0.75, 0.9], 1):
    stop = es.step(vloss)
    print(f'epoch {ep}  val_loss={vloss}  {"→ STOP" if stop else ""}')
    if stop: break
```

검증 손실(val_loss)이 `patience`(참을성) 횟수 이상 연속으로 좋아지지 않으면 학습을 자동으로 멈춥니다. 데모 값을 추적해보면 0.7까지 계속 좋아지다가(wait=0), 0.72부터 나빠지기 시작(wait=1), 0.75도 나쁨(wait=2, patience 도달) → **4번째 epoch에서 멈춥니다.** 계속 학습해봤자 성능이 안 좋아지고 있으니 시간 낭비를 막아줍니다.

## STEP 6 — torch.compile (한 줄 가속)

```python
tp_comp = None
try:
    if IS_CUDA and hasattr(torch, 'compile'):
        base = make_model()
        cmodel = torch.compile(base)
        opt = torch.optim.SGD(cmodel.parameters(), lr=0.01)
        ld = DataLoader(train_ds, batch_size=BATCH, num_workers=best_nw, pin_memory=True)

        # 간단 측정 (compile은 첫 실행에 컴파일 시간이 있어 warmup을 넉넉히)
        def bench_model(m, use_amp=True, steps=STEPS, warmup=8):
            m.train(); scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            it = iter(ld)
            def nxt():
                nonlocal it
                try: return next(it)
                except StopIteration: it = iter(ld); return next(it)
            def stp():
                x, y = nxt(); x, y = x.to(device), y.to(device); opt.zero_grad()
                with torch.autocast(device_type='cuda', enabled=use_amp):
                    loss = crit(m(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); return x.size(0)
            for _ in range(warmup): stp()
            torch.cuda.synchronize(); t0 = time.time(); n = 0
            for _ in range(steps): n += stp()
            torch.cuda.synchronize(); return n / (time.time() - t0)

        tp_comp = bench_model(cmodel)
        SCORE['④ +compile'] = tp_comp
        print(f'torch.compile: {tp_comp:,.0f} img/sec')
    else:
        print('※ CPU이거나 torch.compile 미지원 — 이 단계는 건너뜁니다.')
except Exception as e:
    print('torch.compile 사용 불가(환경 문제) — 건너뜁니다:', str(e)[:80])
```

`torch.compile(model)`은 PyTorch 2.x부터 지원되는 기능으로, 평소 한 줄씩 즉석에서 실행(eager mode)되던 연산 흐름을 **미리 통째로 분석해서 여러 연산을 합치거나(kernel fusion) GPU 최적화 저수준 코드로 미리 컴파일**해둡니다. 처음 컴파일하느라 초반엔 오히려 느릴 수 있어서(그래서 워밍업을 8번으로 넉넉히 잡음), 컴파일 완료 이후부터 매 스텝이 더 빨라집니다. `try/except`로 감싼 이유는 이 기능이 CUDA 환경에서만 지원되고, 환경에 따라 실패할 수 있어서 실패해도 노트북 전체가 멈추지 않게 방어한 것입니다.

## 미니 프로젝트 — 결과 비교

```python
if SCORE:
    ks = list(SCORE.keys()); vs = [SCORE[k] for k in ks]
    plt.figure(figsize=(9, 4.5))
    b = plt.bar(range(len(ks)), vs, color=[GRAY] + [TEAL] * (len(ks) - 1))
    plt.xticks(range(len(ks)), ks, fontsize=9)
    plt.ylabel('처리량 (img/sec)'); plt.title('최적화 단계별 학습 처리량')
    for i, v in enumerate(vs): plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.show()
    print(f'기준선 대비 최고 배속: {max(vs)/vs[0]:.2f}배')
else:
    print('먼저 위 STEP들을 실행해 SCORE를 채우세요.')
```

①기준선 → ②DataLoader → ③AMP → ④compile까지 쌓아온 결과를 막대그래프로 그려서, 각 최적화 단계가 처리량을 얼마나 끌어올렸는지 한눈에 비교합니다.

### 종합문제 (직접 실습)

```python
# 🧪 직접 작성: 배치를 키우며 처리량·메모리 측정
for bs in [32, 64, 128]:
    ld = DataLoader(train_ds, batch_size=bs, num_workers=best_nw, pin_memory=IS_CUDA)
    tp, mem = run_bench(ld, use_amp=True)
    print(f'batch={bs}: {tp:,.0f} img/sec, {mem:.2f} GB')

# 모델을 바꿔 비교하려면: 맨 위 CONFIG에서 MODEL='resnet50' 으로 바꾼 뒤
# STEP 0 데이터/모델 셀을 다시 실행하고 run_bench를 돌려보면 됩니다.
```

---

## 정리

| 기법 | 무엇을 절약/개선하나 |
|---|---|
| `num_workers` + `pin_memory` | 데이터 준비 병목 해소, CPU→GPU 전송 속도 향상 |
| AMP (`autocast`+`GradScaler`) | 연산 속도↑, 메모리↓ (16비트 연산) |
| Gradient Accumulation | 메모리는 적게 쓰면서 큰 배치 효과 |
| `torch.compile` | 연산 그래프 자체를 최적화해 속도↑ |
| 체크포인트(모델+옵티마이저+스케줄러) | 학습 중단 시 정확히 이어서 재개 가능 |
| Early Stopping | 불필요한 학습 시간 절약, 과적합 방지 |
| 시드 고정 | 실험 재현성 확보 (공정 비교의 전제조건) |

이번 CNN 이미지 분류 실습 시리즈를 통해 "정확한 모델을 어떻게 만들까"([1편]({% post_url 2026-09-11-cnn-classification-1-alexnet-transfer-learning %}) 전이학습·파인튜닝, [2편]({% post_url 2026-09-11-cnn-classification-2-vgg13-pretrained-inference %}) 미세조정 없는 추론, [3편]({% post_url 2026-09-11-cnn-classification-3-cnn-scratch-vs-transfer-learning %}) 밑바닥 CNN vs 전이학습)와 "만든 모델을 어떻게 빠르게 학습시킬까"(이번 편의 성능 최적화)라는 CNN 실전 개발의 두 축을 모두 다뤄봤습니다.
