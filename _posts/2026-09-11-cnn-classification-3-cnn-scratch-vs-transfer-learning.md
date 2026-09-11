---
title: "CNN 이미지 분류 실습(3) — 밑바닥부터 쌓은 CNN vs 전이학습(ResNet-18) 성능 비교"
date: 2026-09-11 14:30:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, cnn, resnet, transfer-learning, fine-tuning, cifar-10, batchnorm, dropout, data-augmentation, image-classification, deep-learning]
description: "CIFAR-10으로 CNN을 밑바닥부터 직접 쌓고, BatchNorm·Dropout·데이터 증강을 얹어가며 실험한 뒤, ResNet-18 전이학습(특징추출/미세조정)과 정확도를 한 막대그래프로 비교한다. requires_grad로 백본을 얼리고 푸는 기준까지 코드로 정리했다."
---

> [지난 편]({% post_url 2026-09-11-cnn-classification-2-vgg13-pretrained-inference %})에서는 사전학습 VGG13을 수정 없이 그대로 돌려봤습니다. 이번 편은 반대로 **CNN을 밑바닥부터 직접 만들어보고**, 층을 바꿔가며 실험한 뒤, **전이학습(ResNet-18)과 정확도를 직접 비교**하는 실습입니다. "왜 전이학습이 유리한가"를 숫자로 확인해봅니다. 데이터셋은 CIFAR-10(비행기·자동차·새·고양이 등 10개 클래스, 32×32 이미지)을 사용합니다.

## 전체 그림

```
STEP 0: CIFAR-10 데이터 준비
STEP 1: CNN을 직접 쌓기 (밑바닥부터)
STEP 2: 구조를 바꿔가며 실험 (BatchNorm, Dropout, 층 추가)
STEP 3: 전이학습(ResNet-18)과 성능 비교
STEP 4: 데이터 증강 얹기
STEP 5: 모든 실험 결과 한눈에 비교
종합문제: 커스텀 분류기를 얹은 나만의 최고 모델 만들기
```

![밑바닥부터 쌓은 CNN 여러 구조와 ResNet-18 전이학습(특징추출·미세조정)의 검증 정확도를 막대그래프로 비교한 다이어그램. 전이학습 계열이 밑바닥 CNN보다 높은 정확도를 기록한다](/assets/img/posts/cnn-basics/diagram3-cnn-vs-transfer.svg)

---

## 준비 — 라이브러리 & 설정

```python
import subprocess, os, shutil, time
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)  # 그래프 한글 폰트
shutil.rmtree(os.path.expanduser('~/.cache/matplotlib'), ignore_errors=True)       # 폰트 캐시 초기화

import numpy as np, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
_fp = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(_fp):
    fm.fontManager.addfont(_fp); plt.rcParams['font.family'] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True; plt.rcParams['grid.alpha'] = 0.3
TEAL = '#0D9488'; GRAY = '#94A3B8'; AMBER = '#F59E0B'  # 그래프용 색상 팔레트

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
np.random.seed(42); torch.manual_seed(42)  # 재현성을 위한 시드 고정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CFG: 이 딕셔너리 하나로 전체 실험의 데이터양·속도를 조절함 (빠르게 돌리려면 N_TRAIN, EPOCHS를 줄이면 됨)
CFG = dict(N_TRAIN=3000, N_TEST=1000, EPOCHS=5, BATCH=128)
print('준비 완료 · device:', device, '· CFG:', CFG)
```

`CFG` 딕셔너리 하나로 전체 실험의 데이터양·속도를 조절할 수 있게 설계했습니다. 이 글에 나오는 모든 실험(밑바닥 CNN, 전이학습, 증강)은 이 4개 값(`N_TRAIN`, `N_TEST`, `EPOCHS`, `BATCH`)을 공통으로 씁니다.

## STEP 0 — 데이터 준비 (CIFAR-10, 필요한 만큼만)

`torchvision.CIFAR10`은 전체 170MB를 통째로 받지만, 여기서는 HuggingFace 미러에서 **필요한 장수만 스트리밍**으로 받아 빠르게 시작합니다.

```python
!pip install -q datasets
from datasets import load_dataset

_tr = load_dataset('uoft-cs/cifar10', split='train', streaming=True).shuffle(seed=42, buffer_size=10000)
_te = load_dataset('uoft-cs/cifar10', split='test',  streaming=True)

# train_raw, test_raw: 각각 {'img': PIL.Image, 'label': int} 딕셔너리를 원소로 갖는 파이썬 리스트.
# streaming=True + .take(N) 조합이라, 전체 5만 장을 안 받고 딱 N_TRAIN/N_TEST 장만 내려받는다.
train_raw = list(_tr.take(CFG['N_TRAIN']))
test_raw  = list(_te.take(CFG['N_TEST']))

classes = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
print('학습:', len(train_raw), '· 검증:', len(test_raw))
```

```python
# ── 변환 3종 + Dataset + 로더 ──
C_MEAN, C_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)   # CIFAR 자체 통계 (밑바닥 CNN용)
I_MEAN, I_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)         # ImageNet 통계 (전이학습용)

tf32  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C_MEAN, C_STD)])
aug32 = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), transforms.Normalize(C_MEAN, C_STD)])
tf_tl = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(I_MEAN, I_STD)])

class DS(Dataset):
    # raw: [{'img': PIL.Image, 'label': int}, ...] 형태의 리스트를 감싸는 래퍼.
    # __getitem__은 PyTorch DataLoader 규약에 맞춰 (전처리된 이미지 텐서, 정수 라벨) 튜플을 돌려줌.
    def __init__(s, raw, tf): s.raw, s.tf = raw, tf
    def __len__(s): return len(s.raw)
    def __getitem__(s, i):
        e = s.raw[i]
        return s.tf(e['img'].convert('RGB')), int(e['label'])

def loader(raw, tf, shuffle):
    return DataLoader(DS(raw, tf), batch_size=CFG['BATCH'], shuffle=shuffle)

# tl32/vl32: 밑바닥 CNN용 (원본 32×32, CIFAR 정규화)
tl32, vl32 = loader(train_raw, tf32, True), loader(test_raw, tf32, False)
# tl_aug: STEP 4 증강 실험 전용 학습 로더 (검증은 증강 없이 vl32 그대로 재사용)
tl_aug = loader(train_raw, aug32, True)
# tl_tl/vl_tl: 전이학습용 (64×64로 리사이즈, ImageNet 정규화 — ResNet이 학습됐던 통계에 맞춤)
tl_tl, vl_tl = loader(train_raw, tf_tl, True), loader(test_raw, tf_tl, False)
print('로더 준비 완료')
```

전처리를 **용도별 3가지**로 만듭니다: `tf32`(밑바닥 CNN 평가용, 원본 32px), `aug32`(증강 실험용), `tf_tl`(전이학습용, 64px + ImageNet 정규화값 — ResNet은 ImageNet으로 학습됐으니 그 통계에 맞춰야 함).

```python
# ── 샘플 이미지 8장 (32×32) 확인 ──
def denorm(t, mean, std):
    t = t.clone()
    for c in range(3): t[c] = t[c] * std[c] + mean[c]  # 정규화의 역연산
    return t.clamp(0, 1).permute(1, 2, 0).numpy()        # (C,H,W) → (H,W,C), matplotlib용

xb, yb = next(iter(tl32))  # xb: (batch, 3, 32, 32), yb: (batch,) 정수 라벨
plt.figure(figsize=(12, 1.8))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(denorm(xb[i], C_MEAN, C_STD), interpolation='nearest')
    plt.axis('off'); plt.title(classes[yb[i]], fontsize=10)
plt.suptitle('CIFAR-10 샘플 (원본 32×32)', y=1.12); plt.tight_layout(); plt.show()
```

## 공통 학습/평가 함수

```python
crit = nn.CrossEntropyLoss()
results = {}   # 실험 이름(tag) → 검증 정확도(val accuracy)를 모아두는 점수판

def accuracy(model, ld):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in ld:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t

def fit(model, tl, vl, epochs=None, lr=1e-3, wd=0.0, tag=None):
    epochs = epochs or CFG['EPOCHS']
    model = model.to(device)
    # requires_grad=True인 파라미터만 옵티마이저에 넘김 → 얼린(freeze) 레이어는 자동으로 학습 대상에서 제외됨
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    tr, va = accuracy(model, tl), accuracy(model, vl)
    if tag: results[tag] = va
    print(f'{tag or "model":24s} train={tr:.3f}  val={va:.3f}  ({time.time()-t0:.0f}s)')
    return va

def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

`fit(모델, 학습로더, 검증로더, tag='이름')` 한 줄이면 학습되고, 그 결과가 자동으로 `results` 딕셔너리(`{tag: val_accuracy}`)에 저장되는 구조입니다. 여러 실험을 반복해도 마지막에 한 번에 비교 그래프를 그릴 수 있게 설계되어 있습니다.

## STEP 1 — CNN을 직접 쌓아보기

CNN의 기본 흐름은 **Conv(특징 뽑기) → ReLU → MaxPool(절반으로 줄이기)**를 몇 번 반복한 뒤, **Flatten → Linear(FNN)**으로 분류합니다.

```python
def build_cnn(channels=(32, 64), fc_hidden=(128,), use_bn=False, dropout=0.0):
    layers = []; c = 3   # 입력 채널 3 (RGB)
    for out in channels:                     # ── Conv 블록들 ──
        layers += [nn.Conv2d(c, out, 3, padding=1)]
        if use_bn: layers += [nn.BatchNorm2d(out)]
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        c = out
    layers += [nn.Flatten()]
    size = 32 // (2 ** len(channels))         # MaxPool을 지날 때마다 가로세로가 절반씩 줄어듦 (conv 블록 ≤ 4개 권장)
    feat = c * size * size                    # Flatten 직후 벡터 길이 = 마지막 채널 수 × 세로 × 가로
    for h in fc_hidden:                       # ── FNN(완전연결) 층들 ──
        layers += [nn.Linear(feat, h), nn.ReLU()]
        if dropout > 0: layers += [nn.Dropout(dropout)]
        feat = h
    layers += [nn.Linear(feat, 10)]           # 출력 10클래스
    return nn.Sequential(*layers)

# 기본 CNN: conv 2층 + FNN 1층
cnn = build_cnn(channels=(32, 64), fc_hidden=(128,))
print(cnn)
print('학습 파라미터:', f'{n_params(cnn):,}')
fit(cnn, tl32, vl32, tag='① 기본 CNN (conv2)')
```

`channels`, `fc_hidden`, `use_bn`, `dropout` 인자만 바꾸면 원하는 CNN 구조를 즉석에서 만들 수 있는 **레고 블록형 함수**입니다.

## STEP 2 — 층을 추가·바꾸며 실험

```python
fit(build_cnn((32, 64, 128), (256, 128)),                       tl32, vl32, tag='② FNN 2층')
fit(build_cnn((32, 64, 128), (256,), use_bn=True),               tl32, vl32, tag='③ +BatchNorm')
fit(build_cnn((32, 64, 128), (256,), use_bn=True, dropout=0.3),  tl32, vl32, tag='④ +BN+Dropout')
```

- **BatchNorm**: 각 층의 출력을 정규화해서 학습을 더 안정적이고 빠르게 만듭니다.
- **Dropout**: 학습 중 일부 뉴런을 랜덤하게 꺼서 과적합(overfitting)을 방지합니다.

```python
# 나만의 CNN 구조 실험 예시
my_cnn = build_cnn(channels=(64, 128), fc_hidden=(256,), use_bn=True, dropout=0.3)
print('내 모델 파라미터:', f'{n_params(my_cnn):,}')
fit(my_cnn, tl32, vl32, tag='⑤ 내 CNN')
```

## STEP 3 — 전이학습(ResNet-18)과 비교 ⭐

밑바닥 CNN은 데이터가 적으면 한계가 있습니다. ImageNet으로 미리 학습한 ResNet-18을 빌려오면 훨씬 높은 곳에서 시작할 수 있습니다.

```python
def resnet(mode):
    net = models.resnet18(weights='IMAGENET1K_V1')
    net.fc = nn.Linear(net.fc.in_features, 10)   # 출력을 CIFAR-10의 10개 클래스로 교체
    if mode == 'feature':                                    # 백본 고정, fc만 학습
        for n, p in net.named_parameters(): p.requires_grad = ('fc' in n)
    elif mode == 'finetune':                                 # 뒤쪽(layer4) + fc만 학습
        for n, p in net.named_parameters(): p.requires_grad = ('layer4' in n) or ('fc' in n)
    return net

fit(resnet('feature'),  tl_tl, vl_tl, lr=1e-3, tag='⑥ 전이:특징추출')
fit(resnet('finetune'), tl_tl, vl_tl, lr=1e-4, tag='⑦ 전이:미세조정')
```

- **특징추출(feature extraction) 모드**: `'fc' in n`인 파라미터만 학습 가능(`True`) → **분류기(fc)만 학습**, 나머지 백본 전체는 얼려둡니다.
- **미세조정(fine-tuning) 모드**: `'layer4' in n`도 함께 학습 가능 → 백본의 마지막 블록까지 함께 학습합니다.

> **미세조정은 왜 학습률(`lr`)을 더 작게 쓸까요?** 이미 좋은 특징을 배운 가중치를 큰 학습률로 흔들면 그 지식이 크게 덮어써져 망가질 수 있습니다(**catastrophic forgetting**). 그래서 미세조정은 특징추출(`lr=1e-3`)보다 한 단계 낮은 학습률(`lr=1e-4`)을 씁니다.

> **용어가 이제 정확히 맞아떨어집니다.** [1편]({% post_url 2026-09-11-cnn-classification-1-alexnet-transfer-learning %})에서 짚었듯, 이 블로그는 "전이학습 = 백본 동결 + 마지막 계층만 학습", "파인튜닝 = 더 많은 파라미터를 재학습"으로 구분합니다. 이번 `resnet('feature')`는 `requires_grad`로 백본을 명시적으로 얼렸으니 정의상 **전이학습**이고, `resnet('finetune')`은 `layer4`까지만 풀어 추가로 학습시키는 **부분적인 파인튜닝**에 해당합니다. 1편의 AlexNet 실습은 이런 동결 코드 자체가 없어 `layer1`~`layer4`를 포함한 전체 파라미터가 학습됐으므로, 이번 STEP 3의 `finetune`보다 한 단계 더 나아가 백본 전체를 재학습시킨 경우였던 셈입니다.

## STEP 4 — 증강·정규화로 성능 짜내기

```python
# 증강 로더(tl_aug)로 같은 구조를 다시 학습 → 증강 효과 확인
fit(build_cnn((32, 64, 128), (256,), use_bn=True, dropout=0.3),
    tl_aug, vl32, epochs=CFG['EPOCHS'] + 3, tag='⑧ CNN+증강')
```

같은 구조에 데이터 로더만 `tl_aug`(랜덤 크롭+좌우반전 적용)로 바꿔서 재학습 → 증강이 정확도에 실제로 도움이 되는지 비교합니다.

> **CIFAR-10에서 상하 반전(180° 회전)은 위험합니다.** 좌우반전은 자연스럽지만(자동차가 왼쪽/오른쪽을 보는 건 둘 다 정상), 위아래를 뒤집으면 "거꾸로 된 자동차/동물" 같은 비현실적인 이미지가 되어 학습에 방해가 됩니다.

## STEP 5 — 지금까지 실험을 한눈에 비교

```python
ks = list(results.keys()); vs = [results[k] for k in ks]
colors = [TEAL if ('전이' in k or '증강' in k) else GRAY for k in ks]
plt.figure(figsize=(10, 4.5))
b = plt.bar(range(len(ks)), vs, color=colors)
plt.xticks(range(len(ks)), ks, rotation=35, ha='right', fontsize=9)
plt.ylim(0, 1); plt.ylabel('검증 정확도'); plt.title('내 실험들 — 구조·기법별 정확도')
for i, v in enumerate(vs): plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
plt.tight_layout(); plt.show()

best = max(results, key=results.get)
print(f'최고: {best}  ({results[best]:.3f})')
```

지금까지 `results`에 쌓인 모든 실험(①밑바닥CNN ~ ⑧증강, 전이학습 포함)을 막대그래프로 비교합니다. 전이학습·증강 계열은 청록색으로 강조해서 "밑바닥 CNN보다 전이학습이 얼마나 더 잘하는지" 한눈에 보여줍니다.

## 종합문제 — 커스텀 분류기를 얹은 나만의 최고 모델

```python
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt


class MyResNet(nn.Module):
    """사전학습 ResNet-18을 불러와서 뒤쪽 분류기(FNN)만 바꿔 실험."""
    def __init__(self, fc_hidden=(256,), dropout=0.3, mode='feature', n_classes=10):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        in_feat = backbone.fc.in_features        # 512 — ResNet-18의 GAP 출력 차원
        backbone.fc = nn.Identity()               # 원래 분류기 제거 (통과만 시킴 → backbone(x)는 (batch, 512) 벡터가 됨)
        self.backbone = backbone
        if mode == 'feature':                     # 백본 고정
            for p in self.backbone.parameters(): p.requires_grad = False
        elif mode == 'finetune':                  # 뒤쪽(layer4)만 학습
            for name, p in self.backbone.named_parameters():
                p.requires_grad = ('layer4' in name)
        # 분류기(FNN) — 이 안을 직접 바꿔도 됨
        layers, f = [], in_feat
        for h in fc_hidden:
            layers += [nn.Linear(f, h), nn.ReLU()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            f = h
        layers += [nn.Linear(f, n_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── 학습하며 에폭별 loss·정확도 기록하는 함수 ──
def fit_history(model, tl, vl, epochs=8, lr=1e-3, wd=0.0):
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    # hist: {'train_loss': [...], 'train_acc': [...], 'val_acc': [...]} — 인덱스 i가 (i+1)번째 epoch 결과
    hist = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    def acc(ld):
        model.eval(); c = t = 0
        with torch.no_grad():
            for x, y in ld:
                x, y = x.to(device), y.to(device)
                c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
        return c / t

    for ep in range(epochs):
        model.train(); run_loss = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run_loss += loss.item() * y.size(0); n += y.size(0)
        tr_loss = run_loss / n
        tr_acc, va_acc = acc(tl), acc(vl)
        hist['train_loss'].append(tr_loss)
        hist['train_acc'].append(tr_acc)
        hist['val_acc'].append(va_acc)
        print(f'epoch {ep+1}/{epochs}  loss={tr_loss:.3f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}')
    return hist


# ── 학습 실행 (전이학습이라 tl_tl / vl_tl 사용) ──
model = MyResNet(fc_hidden=(256,), dropout=0.3, mode='feature')
print('학습 파라미터:', f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
hist = fit_history(model, tl_tl, vl_tl, epochs=8, lr=1e-3)


# ── 그래프: 왼쪽=Loss, 오른쪽=정확도 ──
ep = range(1, len(hist['train_loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(ep, hist['train_loss'], 'o-', color='#B91C1C')
ax1.set_title('학습 Loss'); ax1.set_xlabel('에폭'); ax1.set_ylabel('loss'); ax1.grid(alpha=0.3)

ax2.plot(ep, hist['train_acc'], 'o-', color='#94A3B8', label='train')
ax2.plot(ep, hist['val_acc'],   'o-', color='#0D9488', label='val')
ax2.set_ylim(0, 1); ax2.set_title('정확도'); ax2.set_xlabel('에폭'); ax2.set_ylabel('accuracy')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()
print(f'최종 val 정확도: {hist["val_acc"][-1]:.3f}')
```

`nn.Identity()`(아무 것도 하지 않는 통과 레이어)로 ResNet의 기존 fc를 통째로 제거하고, 그 자리에 원하는 만큼의 층으로 **새 분류기를 직접 설계**해 붙였습니다. `fit_history()`는 에폭마다 loss/정확도를 기록해서, 학습 과정 전체를 곡선으로 볼 수 있게 해줍니다 → 과적합(train은 계속 오르는데 val은 정체) 여부를 진단하는 용도입니다.

---

## 정리

1. **`build_cnn()` 패턴**: 함수 인자로 구조를 조절해서 여러 실험을 빠르게 반복하는 설계
2. **특징추출 vs 미세조정**: `requires_grad`로 어느 레이어를 얼릴지/풀지 결정하고, 미세조정은 학습률을 작게 써야 기존 지식이 안 망가짐
3. **밑바닥 CNN < 전이학습**: 데이터가 적을 때(3000장)는 직접 만든 CNN보다, 이미 학습된 ResNet을 가져다 쓰는 게 압도적으로 유리함

이 시리즈는 다음 편에서 계속됩니다.
