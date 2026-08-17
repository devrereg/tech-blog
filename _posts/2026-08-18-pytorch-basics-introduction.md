---
title: "PyTorch 기본 소개 — 특징과 사용법 완전 정리"
date: 2026-08-18 09:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, python, tensor, autograd]
description: "Tensor를 만들고, 자동미분을 이해하고, 모델 하나를 직접 학습시키는 전 과정을 PyTorch 기준으로 정리했다."
---

> Tensor를 만들고, 자동미분을 이해하고, 모델 하나를 직접 학습시키는 전 과정을 정리한 글입니다.

## 📌 목차

1. 딥러닝 프레임워크란, 그리고 PyTorch
2. 설치와 import
3. PyTorch 핵심 4요소 — 전체 지도
4. 텐서(Tensor) 완전 정복
5. autograd — 자동미분
6. nn.Module — 모델을 클래스로
7. 손실함수와 옵티마이저
8. 학습 사이클 — 5단계 반복
9. 학습 결과 확인하기
10. GPU에서 학습하기 — 정리 패턴

---

## 1. 딥러닝 프레임워크란, 그리고 PyTorch

**프레임워크**는 텐서 연산·자동미분·신경망 계층·최적화를 미리 구현해 둔 도구 모음입니다. 우리는 "무엇을 학습할지"만 설계하면 됩니다.

**PyTorch**는 Meta(구 Facebook) AI 연구소가 공개한 오픈소스 딥러닝 프레임워크로, 연구·논문 구현의 사실상 표준입니다.

### 핵심 특징 두 가지

- **동적 계산 그래프 (define-by-run)**: 코드를 실행하는 순간 계산 그래프가 만들어집니다. 디버깅이 쉽고, `if`/`for` 같은 일반 제어문을 자유롭게 쓸 수 있습니다.
- **GPU 가속**: 텐서에 `.to('cuda')` 한 줄만 붙이면 GPU로 옮겨져 CPU 대비 수십~수백 배 빠른 병렬 연산이 가능합니다.

> **한 문장 요약**: PyTorch = 텐서 + 자동미분 + 신경망 도구

### PyTorch vs TensorFlow

| 구분 | PyTorch | TensorFlow / Keras |
|---|---|---|
| 계산 그래프 | 동적(실행하며 생성) → 디버깅 쉬움 | 정적 위주(과거), 2.x부터 동적 지원 |
| 문법 스타일 | 파이썬·넘파이와 유사, 직관적 | 고수준 Keras API로 간결 |
| 주 사용처 | 연구·논문·프로토타이핑 강세 | 산업·모바일·서빙 생태계 강세 |
| 학습 난이도 | 파이썬 사용자에게 친숙 | Keras는 쉬우나 내부는 복잡 |

연구·실험 단계는 PyTorch, 대규모 서빙은 TensorFlow가 강세를 보이는 것이 대략적인 업계 구도입니다.

---

## 2. 설치와 import

```python
# Colab에는 PyTorch가 이미 설치되어 있다
# 로컬 설치는 pytorch.org 의 안내 명령 사용
# 예) pip install torch torchvision

import torch
import torch.nn as nn         # 신경망 계층
import torch.optim as optim   # 옵티마이저
print(torch.__version__)      # 버전 확인
```

**GPU 사용 가능 여부 점검:**

```python
torch.cuda.is_available()      # GPU 사용 가능? True/False
torch.cuda.get_device_name(0)  # GPU 이름
```

### PyTorch 생태계

| 패키지 | 역할 |
|---|---|
| `torch` | 텐서·자동미분·연산의 핵심 |
| `torchvision` | 이미지 데이터셋·전처리·사전학습 모델 |
| `torchaudio` | 오디오 데이터·변환 |
| `torch.nn` / `optim` | 신경망 계층과 최적화 알고리즘 |

---

## 3. PyTorch 핵심 4요소 — 전체 지도

오늘 다룰 내용을 미리 정리하면 다음과 같습니다.

![PyTorch 핵심 요소 파이프라인](/assets/img/posts/deep-learning-pytorch/05_pipeline.png)

| # | 요소 | 설명 |
|---|---|---|
| 1 | **Tensor** | 다차원 배열. 모든 데이터·파라미터의 그릇 |
| 2 | **autograd** | `loss.backward()` 한 줄로 기울기를 자동 계산 |
| 3 | **nn.Module** | 신경망(모델)을 정의하는 기본 클래스 |
| 4 | **Optimizer** | 기울기로 가중치를 업데이트 — SGD·Adam |

전체 흐름을 한 문장으로 정리하면: **데이터를 Tensor로 만들고 → nn.Module로 만든 모델에 통과시켜 → autograd로 기울기를 계산하고 → Optimizer로 가중치를 업데이트**하는 사이클입니다.

---

## 4. 텐서(Tensor) 완전 정복

### 4.1 텐서란? — 차원으로 이해하기

텐서는 "숫자들의 다차원 배열"입니다. 넘파이 `ndarray`와 거의 같지만, **GPU 연산**과 **자동미분**이 가능하다는 점이 다릅니다.

![텐서 차원 시각화](/assets/img/posts/deep-learning-pytorch/02_tensor_dims.png)

| 차원 | 이름 | 예시 |
|---|---|---|
| 0차원 | 스칼라 | `3.14` |
| 1차원 | 벡터 | `[1, 2, 3]` |
| 2차원 | 행렬 | `shape (2,3)` |
| 3차원 이상 | 고차원 텐서 | `[B, C, H, W]` (이미지 배치) |

이미지·문장·소리 등 모든 입력 데이터와 모델의 가중치가 전부 텐서로 표현됩니다.

### 4.2 텐서 만들기

**① 값으로 직접 (상수)**

```python
a = torch.tensor(3.0)              # 스칼라(0D)
b = torch.tensor([1, 2, 3])        # 벡터(1D)
c = torch.tensor([[1., 2.],
                   [3., 4.]])      # 행렬(2D)

print(c.shape)   # torch.Size([2, 2])
```

> 정수(`1`)는 `int64`, 실수(`1.`)는 `float32`가 됩니다. 학습엔 보통 실수형을 씁니다.

**② 패턴 생성 함수**

```python
torch.zeros(2, 3)     # 0으로 채운 2x3
torch.ones(3)         # 1로 채운 길이 3 벡터
torch.full((2,2), 7)  # 7로 채운 2x2
torch.eye(3)          # 3x3 단위행렬

torch.arange(0, 10, 2)     # tensor([0, 2, 4, 6, 8])
torch.linspace(0, 1, 5)    # tensor([0.00,0.25,0.50,0.75,1.00])

x = torch.ones(2, 3)
torch.zeros_like(x)   # x와 같은 모양의 0텐서
```

**③ 랜덤과 재현성**

```python
torch.manual_seed(42)   # 재현성: 같은 난수 보장

torch.rand(2, 3)        # [0,1) 균등분포
torch.randn(2, 3)       # 표준정규분포(평균0, 표준편차1)
torch.randint(0, 10, (2, 3))  # 0~9 정수
```

> 신경망 가중치는 보통 `randn` 계열로 초기화되며, `manual_seed`를 고정하면 실행할 때마다 같은 결과를 재현할 수 있습니다.

### 4.3 텐서의 속성

```python
x = torch.randn(2, 3)

x.shape     # torch.Size([2, 3]) — 모양
x.ndim      # 2 — 차원 수
x.dtype     # torch.float32 — 자료형
x.device    # cpu — 어디에 있나
x.numel()   # 6 — 전체 원소 개수
```

> **shape · dtype · device** 이 세 속성이 디버깅의 90%를 차지합니다. 에러가 나면 이 셋부터 확인하세요.

### 4.4 자료형(dtype)과 형변환

```python
x = torch.tensor([1, 2, 3])   # int64

x.float()                      # float32로 변환
x.to(torch.float32)            # 동일

(x > 1)   # tensor([False, True, True])  — 불리언 마스킹
```

> **분류 라벨은 `long`, 나머지는 `float`**만 기억해도 초반 에러의 절반은 사라집니다.

### 4.5 인덱싱과 슬라이싱

```python
x = torch.tensor([[10, 11, 12],
                   [20, 21, 22],
                   [30, 31, 32]])

x[0]       # 첫 행 → [10, 11, 12]
x[0, 2]    # 0행 2열 → 12
x[:, 1]    # 모든 행의 1열 → [11, 21, 31]
x[1:, :2]  # 1행부터, 0~1열 → [[20,21],[30,31]]

x[x > 20]  # 조건(불리언) 인덱싱 → [21, 22, 30, 31, 32]
```

> ⚠️ 슬라이싱은 '복사'가 아니라 '뷰(view)'인 경우가 많아 원본과 메모리를 공유합니다. 진짜 복사본이 필요하면 `.clone()`을 씁니다.

### 4.6 모양 바꾸기 — reshape · view · squeeze

```python
x = torch.arange(6)    # [0,1,2,3,4,5]

x.reshape(2, 3)         # 2x3로
x.view(2, 3)            # reshape과 거의 동일 (연속 메모리)
x.reshape(-1, 2)        # -1 = 나머지 축 자동 계산

y = torch.zeros(1, 3, 1)
y.squeeze().shape        # [3]  (크기 1 축 제거)
y.unsqueeze(0).shape     # [1, 1, 3, 1] (축 추가)
torch.flatten(x)          # 1D로 펼치기
```

> 변환 전후 전체 원소 개수(`numel`)는 반드시 같아야 합니다. MNIST에서 `[B,1,28,28]` 이미지를 `[B,784]`로 펴는 Flatten이 바로 이 reshape입니다.

### 4.7 사칙연산과 브로드캐스팅

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([10., 20., 30.])

a + b     # [11, 22, 33]  (원소별)
a * b     # [10, 40, 90]
a ** 2    # [1, 4, 9]

# 브로드캐스팅: 모양이 달라도 자동 확장
a + 100   # [101, 102, 103]  (스칼라)

m = torch.ones(2, 3)
m + torch.tensor([1., 2., 3.])   # 행마다 더함
```

> **브로드캐스팅 규칙**: 뒤축부터 비교해서, 크기가 같거나 둘 중 하나가 1이면 연산이 가능합니다. 이 덕분에 '편향(bias)을 모든 샘플에 더하기' 같은 연산을 반복문 없이 한 줄로 처리합니다.

### 4.8 행렬곱과 집계 연산

```python
A = torch.tensor([[1., 2.], [3., 4.]])
x = torch.tensor([1., 1.])

A @ x               # 행렬곱 → [3., 7.]
torch.matmul(A, x)   # 동일

t = torch.tensor([[1., 2.], [3., 4.]])
t.sum()          # 10.
t.mean()         # 2.5
t.max()          # 4.
t.sum(dim=0)      # 열 방향 합 → [4., 6.]
t.argmax()        # 최댓값 위치(인덱스)
```

> 신경망 한 층의 핵심 계산이 바로 행렬곱(`Wx+b`)입니다. `argmax`는 분류 결과를 뽑을 때 다시 만납니다.

### 4.9 NumPy ↔ Tensor 변환

```python
import numpy as np

arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)   # NumPy → Tensor
back = t.numpy()             # Tensor → NumPy

# 둘은 메모리를 공유! 하나를 바꾸면 둘 다 바뀐다
arr[0] = 99   # t[0]도 99로 바뀐다
```

> matplotlib으로 그림을 그릴 땐 `.detach().cpu().numpy()` 순서로 변환합니다 (자동미분·GPU 연결 해제).

### 4.10 CPU ↔ GPU 이동

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.randn(3).to(device)      # 텐서 이동
model = model.to(device)            # 모델도 같은 장치로!

# 연산하려면 둘 다 같은 장치에 있어야 한다
```

> 흔한 에러: `"Expected all tensors to be on the same device"` → 입력 또는 모델 중 하나에 `.to(device)`를 빠뜨린 것입니다.

---

## 5. autograd — 자동미분

### 5.1 상수 vs 학습되는 변수

```python
a = torch.tensor(2.0)
a.requires_grad     # False — 일반 텐서(상수)

w = torch.tensor(2.0, requires_grad=True)
w.requires_grad     # True — 학습되는 변수
```

데이터(x, y)는 상수, 가중치(w, b)는 학습 변수입니다. **학습이란 결국 `requires_grad=True`인 변수들을 손실이 작아지는 방향으로 조금씩 바꾸는 일**입니다.

### 5.2 backward() — 기울기 자동 계산

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)

y = w * x + b        # 순전파 (계산 그래프 생성)
L = (y - 10) ** 2      # 손실

L.backward()            # 역전파! 기울기 자동 계산
print(w.grad)           # dL/dw
print(b.grad)           # dL/db
```

- **순전파**: 연산을 따라가며 계산 그래프를 자동 기록
- **backward()**: 그래프를 거꾸로 따라가며 연쇄법칙(chain rule) 적용 — 역전파
- **.grad**: 각 변수에 `∂L/∂변수` 값이 채워짐

### 5.3 손계산으로 검증 — y = x²

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # tensor(4.)

# 수학적으로 dy/dx = 2x, x=2 → 4  ✓
```

autograd 결과와 손으로 구한 미분이 정확히 일치합니다 — **마법이 아니라 미분**입니다.

### 5.4 기울기는 누적된다 — zero_grad의 이유

```python
w = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    L = (w * 2) ** 2
    L.backward()
    print(w.grad)   # 8, 16, 24 ...  계속 더해짐!

# → 매 스텝 시작에 기울기를 0으로 비워야 한다
w.grad.zero_()          # 또는 optimizer.zero_grad()
```

`.grad`의 기본 동작은 "덮어쓰기"가 아니라 "누적"입니다. RNN 등에서 기울기를 모아야 할 때가 있어 이렇게 설계되었지만, 일반적인 학습 루프에서는 매번 직접 비워줘야 합니다.

> **비우고(zero_grad) → 계산(backward) → 적용(step)** — 이 순서가 학습 루프의 뼈대입니다.

### 5.5 추론할 땐 그래프를 끈다 — no_grad

```python
with torch.no_grad():
    pred = model(x)   # 그래프 추적 안 함 → 속도↑, 메모리↓

# 특정 텐서 하나만 그래프에서 분리
v = y.detach()
v = y.detach().cpu().numpy()   # 시각화용
```

학습이 끝난 뒤 예측만 할 때는 `.backward()`를 호출할 일이 없으므로, `no_grad()`로 그래프 자체를 만들지 않게 하면 빠르고 안전합니다.

---

## 6. nn.Module — 모델을 클래스로

```python
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)   # w, b 자동 생성·추적

    def forward(self, x):
        return self.fc(x)            # 순전파 정의

model = LinearModel()
list(model.parameters())   # 학습될 w, b
```

| 부분 | 역할 |
|---|---|
| `class Model(nn.Module)` | 모든 PyTorch 모델의 기본 뼈대 |
| `__init__` | 사용할 계층(layer)을 선언 |
| `forward(self, x)` | 데이터가 흐르는 길을 정의 |
| `model.parameters()` | 학습 대상(w, b 등)을 모아서 반환 |

직접 `w, b`를 들고 다닐 필요 없이, `nn.Linear`가 파라미터를 자동 생성·추적합니다. 계층이 여러 개인 MLP·CNN도 이 틀(`__init__` + `forward`)을 그대로 씁니다.

---

## 7. 손실함수와 옵티마이저

```python
criterion = nn.MSELoss()          # 회귀용 손실
# nn.CrossEntropyLoss()            # 분류용 손실

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01)
# torch.optim.Adam(model.parameters(), lr=0.01)
```

| 손실함수 | 용도 |
|---|---|
| `nn.MSELoss()` | 회귀(연속값 예측) |
| `nn.CrossEntropyLoss()` | 분류(클래스 예측, 예: MNIST) |

| 옵티마이저 | 특징 |
|---|---|
| `SGD` | 기본 경사하강법 |
| `Adam` | 학습률 자동 조절 — 더 빠르고 안정적, 입문자에게 추천 |

---

## 8. 학습 사이클 — 5단계 반복

지금까지 배운 조각을 모두 조립하면 다음과 같은 순환 구조가 완성됩니다.

![학습 사이클 5단계](/assets/img/posts/deep-learning-pytorch/01_training_cycle.png)

| 단계 | 하는 일 |
|---|---|
| ① zero_grad | 기울기 비우기 (누적 방지) |
| ② forward | 현재 가중치로 예측 만들기 |
| ③ loss | 예측과 정답의 오차 측정 |
| ④ backward | 기울기 자동 계산 |
| ⑤ step | 새 가중치 = 기존 - 학습률×기울기 |

### 완성된 학습 루프 — 표준 5줄

```python
# 데이터: y = 2x + 1 (숨겨진 정답)
X = torch.tensor([[1.], [2.], [3.], [4.]])   # shape (4, 1)
y = torch.tensor([[3.], [5.], [7.], [9.]])

model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()      # ① 비우기
    pred = model(X)              # ② 순전파
    loss = criterion(pred, y)   # ③ 손실
    loss.backward()              # ④ 역전파
    optimizer.step()             # ⑤ 갱신
```

> **이 5줄이 딥러닝 학습 전체의 공통 골격입니다.** 모델 구조와 손실함수, 옵티마이저 종류만 바뀔 뿐, MLP든 CNN이든 트랜스포머든 이 루프 자체는 변하지 않습니다.

`nn.Linear(1,1)`은 `(배치크기, 입력차원)` 형태를 기대하므로, 데이터를 1D `[4]`가 아닌 2D 열벡터 `[4, 1]`로 준비해야 합니다.

---

## 9. 학습 결과 확인하기

200번의 반복 끝에 손실이 실제로 줄어들고, 모델이 숨겨진 규칙을 찾아내는지 확인해봅니다.

![학습 손실 곡선](/assets/img/posts/deep-learning-pytorch/03_loss_curve.png)

- **초반(epoch 0~25)**: 시작점이 정답과 많이 떨어져 있어 기울기가 크고, 손실이 빠르게 감소
- **후반(epoch 100 이후)**: 정답에 가까워질수록 기울기가 작아지며 완만하게 수렴

```python
# 학습된 파라미터 확인
print(model.fc.weight.item())   # ≈ 2.0
print(model.fc.bias.item())     # ≈ 1.0

# 학습에 없던 새 입력으로 예측
with torch.no_grad():
    print(model(torch.tensor([[5.]])))  # ≈ 11
```

![회귀 결과 시각화](/assets/img/posts/deep-learning-pytorch/04_regression_result.png)

`weight≈2, bias≈1`로 수렴했고, 학습 데이터(`x=1~4`)에 없던 `x=5`도 `y≈11`로 정확히 예측했습니다. 이는 모델이 데이터를 단순히 암기한 게 아니라, **`y = 2x + 1`이라는 규칙 자체를 학습했다는 증거**입니다(일반화, generalization).

---

## 10. GPU에서 학습하기 — 정리 패턴

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)   # 모델 이동

for epoch in range(200):
    X, y = X.to(device), y.to(device)   # 데이터 이동
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

GPU를 쓴다고 학습 로직이 바뀌지 않습니다. **모델·데이터를 같은 device로만 옮기면**, 학습 루프의 나머지 코드는 완전히 동일합니다. 오늘 다룬 모델(`nn.Linear(1,1)`)은 파라미터가 2개뿐이라 GPU 효과가 크지 않지만, MNIST처럼 큰 모델일수록 GPU 가속 체감이 커집니다.

---

## 정리

| 배운 내용 | 핵심 키워드 |
|---|---|
| 프레임워크·PyTorch 소개 | 동적 계산 그래프, GPU 가속 |
| Tensor | 생성, shape/dtype/device, reshape, 연산, 넘파이 변환 |
| autograd | requires_grad, backward(), zero_grad, no_grad |
| nn.Module | 모델 클래스화, parameters() |
| 손실·옵티마이저 | criterion, optimizer(SGD/Adam) |
| 학습 루프 | zero_grad → forward → loss → backward → step |
| 결과 검증 | 파라미터 수렴, 새 입력 예측(일반화) |

텐서를 만들고 → 자동미분을 이해하고 → 모델 하나를 직접 학습시키는 전 과정을 손에 익히는 것이 오늘의 목표였습니다. 이 5줄짜리 학습 루프는 앞으로 배울 MLP, CNN, Transformer 등 어떤 모델에도 그대로 적용되는 PyTorch의 핵심 골격입니다.
