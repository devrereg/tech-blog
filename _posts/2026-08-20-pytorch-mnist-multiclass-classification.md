---
title: "PyTorch로 배우는 다중분류 완전정복: MNIST부터 모델 평가지표까지"
date: 2026-08-20 09:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, mnist, multi-class-classification, machine-learning]
description: "MNIST 손글씨 숫자 데이터셋으로 다중분류 신경망을 만들고, 혼동행렬·Precision/Recall·ROC-AUC 같은 실전 모델 평가지표까지 처음부터 끝까지 정리했다."
---

> 이 글은 MNIST 손글씨 숫자 데이터셋을 활용한 다중분류(Multi-class Classification) 수업 내용을 정리한 포스트입니다. 신경망의 기본 구조부터 시작해, 실전에서 반드시 알아야 할 모델 평가 지표(혼동행렬, Precision/Recall, ROC-AUC)까지 전체 흐름을 다룹니다.

## 📌 목차

1. [문제 정의: MNIST 손글씨 숫자 인식](#1-problem-definition)
2. [컴퓨터는 이미지를 어떻게 이해하는가](#2-image-as-numbers)
3. [신경망 구조: 입력층 · 은닉층 · 출력층](#3-network-architecture)
4. [ReLU 활성화 함수](#4-relu)
5. [GPU와 미니 배치 학습](#5-gpu-minibatch)
6. [데이터 파이프라인: Dataset · Transforms · DataLoader](#6-data-pipeline)
7. [모델 정의와 학습 결과](#7-model-training)
8. [경사 소실 문제 · 람다 표현식 · 배치 사이즈](#8-vanishing-gradient)
9. [혼동행렬(Confusion Matrix)](#9-confusion-matrix)
10. [Precision과 Recall, 그리고 F1-Score](#10-precision-recall-f1)
11. [다중 클래스 평가: 평균 방식 비교](#11-multiclass-averaging)
12. [ROC Curve와 AUC](#12-roc-auc)
13. [확률 보정(Calibration)](#13-calibration)
14. [클래스 불균형(Class Imbalance) 해결 전략](#14-class-imbalance)
15. [전체 정리](#15-summary)

---

## 1. 문제 정의: MNIST 손글씨 숫자 인식 {#1-problem-definition}

지금까지 정형화된 "숫자" 데이터를 다뤘다면, 이번엔 한 단계 나아가 모델이 **이미지를 직접 "보고 이해"** 하도록 학습시킨다. 목표는 손으로 쓴 0~9 숫자 이미지를 보고 어떤 숫자인지 맞히는 것이다.

사람은 숫자를 픽셀 값이 아니라 **모양(형태)** 으로 인식한다. 이런 시각 정보 이해 기술은 자율주행차의 사물 인식, 의료 AI의 질병 진단 등 실생활 AI 기술의 핵심 원리이기도 하다.

## 2. 컴퓨터는 이미지를 어떻게 이해하는가 {#2-image-as-numbers}

컴퓨터에게 이미지란 숫자로 가득 찬 격자판일 뿐이다.

- MNIST 이미지는 **28 × 28 = 784개 픽셀**로 구성
- 각 픽셀은 **0(검은색) ~ 255(흰색)** 사이의 밝기 값을 가짐

이 원본 데이터를 신경망에 입력하려면 벡터화 과정을 거쳐야 한다.

| 단계 | 값의 범위 | 형태(shape) |
|---|---|---|
| 원본 이미지 | 0 ~ 255 | 28 × 28 |
| PyTorch 텐서화 (ToTensor) | 0 ~ 1 | [1, 28, 28] |
| 완전결합 신경망 입력용 펼치기 | 0 ~ 1 | [784] |

완전결합 신경망(Fully Connected Network)은 1차원 벡터 입력을 요구하기 때문에, 2차원 이미지를 784개의 숫자가 일렬로 늘어선 벡터로 펴주는 전처리가 반드시 필요하다.

## 3. 신경망 구조: 입력층 · 은닉층 · 출력층 {#3-network-architecture}

다중분류 신경망은 크게 3단계로 구성된다.

![MNIST 다중분류 신경망 구조](/assets/img/posts/mnist-images/01_nn_architecture.png)

| 층 | 뉴런 수 | 역할 |
|---|---|---|
| **입력층** | 784개 | 원본 이미지 데이터를 받아들이는 첫 관문 |
| **은닉층** | 128개 | 입력에서 직접 보이지 않는 특징·패턴을 학습하는 중간 계산 단계 |
| **출력층** | 10개 | 0~9 중 어느 숫자인지 최종 예측 |

예를 들어 숫자 '8'을 인식할 때, 은닉층은 "상단 동그라미", "하단 동그라미" 같은 중간 특징을 학습한다. 이렇게 여러 층을 쌓아 복잡한 문제를 단계적으로 해결하는 것이 **딥러닝(Deep Learning)** 의 핵심이다.

## 4. ReLU 활성화 함수 {#4-relu}

### 비선형성이 필요한 이유

선형 함수만 여러 겹 쌓으면 수학적으로 결국 하나의 큰 선형 함수와 동일해진다. 층을 깊게 쌓는 의미를 가지려면 각 층의 결과에 **비선형 함수(활성화 함수)**를 적용해야 한다.

### ReLU의 동작 원리

$$ReLU(x) = \max(0, x)$$

- 입력값이 0보다 작으면 → 0 출력
- 입력값이 0보다 크면 → 입력값 그대로 출력

![ReLU 활성화 함수 그래프](/assets/img/posts/mnist-images/02_relu_function.png)

그래프에서 x=0 지점에 생기는 '꺾임'이 바로 비선형성의 정체다. 이 덕분에 여러 층을 쌓았을 때 신경망이 복잡하고 구불구불한 패턴까지 표현할 수 있다.

```python
relu = nn.ReLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = relu(x)  # tensor([0., 0., 0., 1., 2.])
```

## 5. GPU와 미니 배치 학습 {#5-gpu-minibatch}

### GPU가 필요한 이유

MNIST 훈련 데이터는 60,000장에 달한다. 이 방대한 연산을 CPU로 처리하면 시간이 오래 걸리지만, GPU는 병렬 연산에 특화되어 있어 학습 속도를 폭발적으로 향상시킨다.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

> PyTorch에서는 모든 텐서와 모델이 자신이 어느 장치에 속해 있는지 정보를 가지며, **서로 다른 장치에 있는 텐서끼리는 연산할 수 없다.** 데이터와 모델을 모두 `.to(device)`로 같은 장치에 위치시켜야 한다.

### 미니 배치(Mini-Batch)

60,000개를 한 번에 학습시키는 것은 비효율적이다. 데이터를 작은 묶음(예: 500개씩)으로 나눠 반복 학습하며, 이 묶음 하나를 **미니 배치**라고 부른다.

## 6. 데이터 파이프라인: Dataset · Transforms · DataLoader {#6-data-pipeline}

PyTorch의 데이터 준비 도구는 전문 셰프의 주방 도구에 비유할 수 있다.

![데이터 전처리 파이프라인](/assets/img/posts/mnist-images/06_data_pipeline.png)

| 도구 | 역할 | 비유 |
|---|---|---|
| **Dataset** | 원본 데이터(이미지 파일)를 인덱스로 접근 가능하게 제공 | 창고에서 재료 꺼내기 |
| **Transforms** | 텐서 변환, 정규화, 모양 변경 등 전처리 수행 | 재료 씻고 다듬기 |
| **DataLoader** | 전체 데이터를 `batch_size`만큼 묶어 미니 배치 생성 | 1인분씩 소분하기 |

### Transforms의 4가지 주요 기능

1. **`ToTensor()`**: PIL 이미지 → 텐서 변환, 값 범위 [0,255] → [0,1]
2. **`Normalize(mean, std)`**: 예) `Normalize(0.5, 0.5)` 적용 시 [0,1] → [-1,1]
3. **`Lambda(lambda x: ...)`**: 사용자 정의 변환(예: `x.view(-1)`로 펼치기)
4. **`Compose([...])`**: 여러 변환을 순서대로 묶어 하나의 파이프라인으로 구성

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.Lambda(lambda x: x.view(-1)),
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

batch_size = 500
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

**결과:** 60,000 ÷ 500 = **120개**의 미니 배치, `images.shape = [500, 784]`, `labels.shape = [500]`

> `shuffle=True`는 매 epoch마다 데이터 순서를 무작위로 섞어, 모델이 데이터 순서 자체를 외우는 것을 방지하고 일반화 성능을 높인다. 검증 데이터는 학습에 관여하지 않으므로 섞을 필요가 없다(`shuffle=False`).

## 7. 모델 정의와 학습 결과 {#7-model-training}

### nn.Module로 신경망 정의하기

```python
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3
```

- `__init__`: 필요한 층(도구)을 미리 준비
- `forward`: 데이터가 실제로 흘러가는 순서(l1 → relu → l2)를 정의

여기서 눈여겨볼 점은 `forward`의 마지막 출력(`x3`)에 **softmax를 적용하지 않고 로짓(logit)을 그대로 반환**한다는 것이다. 다중 분류에서 흔히 쓰는 손실 함수인 `nn.CrossEntropyLoss`가 내부적으로 `log-softmax`와 `NLLLoss`를 함께 수행하기 때문에, 모델이 직접 softmax까지 계산해서 넘겨주면 오히려 이중으로 적용되어 학습이 잘못된다. 그래서 모델은 각 클래스에 대한 점수(로짓)만 계산하고, 확률로 변환하는 일은 손실 함수에게 맡기는 것이 PyTorch의 표준 설계다.

### 손실 함수 · 옵티마이저 · 학습 루프

모델과 데이터 파이프라인이 준비되었다면, 실제로 가중치를 업데이트하는 학습 루프를 작성한다.

```python
model = Net(n_input=784, n_output=10, n_hidden=128).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
```

- `optimizer.zero_grad()`: 이전 배치에서 계산된 기울기가 누적되지 않도록 초기화
- `loss.backward()`: 역전파로 각 파라미터의 기울기 계산
- `optimizer.step()`: 계산된 기울기로 가중치 업데이트

이 4줄(`zero_grad → forward → backward → step`)이 미니 배치 하나를 학습하는 기본 사이클이며, 이를 전체 배치(120개)와 전체 epoch(100번)만큼 반복하는 것이 학습 루프의 전부다.

### 학습 결과

| 항목 | 값 |
|---|---|
| 훈련 데이터 | 60,000장 |
| 학습 반복 횟수(Epoch) | 100 |
| 최종 정확도(검증 기준) | 94.95% |

(실제로 재현한 실험 결과다.) 은닉층을 1개 더 추가(2개 은닉층 구조)한 결과, 정확도가 **94.95% → 95.81%**로 약 1%p 향상되었다. PyTorch는 모델 구조를 유연하게 바꿔가며 실험하기 좋다는 것을 보여주는 결과다.

## 8. 경사 소실 문제 · 람다 표현식 · 배치 사이즈 {#8-vanishing-gradient}

### 경사 소실 문제(Vanishing Gradient)와 ReLU

시그모이드 함수는 미분값이 최대 0.25에 불과해, 층이 깊어질수록 역전파 과정에서 이 값들이 반복적으로 곱해지며 경사가 0에 가깝게 소실된다. 반면 ReLU는 입력이 양수일 때 미분값이 항상 1이므로 경사가 줄어들지 않고 초반 층까지 효율적으로 전달된다.

실제로 재현한 실험 결과, 동일 조건에서 첫 번째 층의 경사 값 평균은:

| 활성화 함수 | 경사 값 평균 |
|---|---|
| 시그모이드 | 0.000175 |
| ReLU | 0.000848 (약 5배 큼) |

### 람다(Lambda) 표현식

이름 없이 즉석에서 만드는 함수. `transforms.Compose` 안에서 텐서 모양을 바꾸는 것처럼 간단한 처리에 유용하다.

```python
# 일반 함수
def flatten(x):
    return x.view(-1)

# 람다 표현식 (더 간결)
flatten = lambda x: x.view(-1)
```

### 배치 사이즈의 중요성

| 배치 사이즈 | 특징 | 정확도 경향 |
|---|---|---|
| 크다 (예: 500) | 안정적, 계산 빠름 | 상대적으로 낮음 |
| 작다 (예: 50) | 노이즈로 인해 국소 최적점 탈출에 유리 | 상대적으로 높음 |

작은 배치 사이즈의 '노이즈'가 모델이 국소 최적점(Local Minimum)에서 벗어나 더 좋은 지점을 탐색하도록 돕는다.

---

## 9. 혼동행렬(Confusion Matrix) {#9-confusion-matrix}

7장에서 학습시킨 MNIST 모델은 검증 정확도 94.95%를 기록했다. 하지만 정확도라는 숫자 하나만으로는 이 모델이 **어떤 종류의 실수**를 하는지 알 수 없다 — 예를 들어 숫자 4를 9로 착각하는지, 3을 5로 착각하는지는 정확도에 전혀 드러나지 않는다. 이럴 때 필요한 도구가 혼동행렬이다. 혼동행렬은 예측 결과를 실제 정답과 비교해 표로 정리한 도구로, MNIST처럼 클래스가 10개(0~9)인 경우 10×10 크기의 표가 된다.

### 이진분류 기본 4분면

| | 예측: 긍정 | 예측: 부정 |
|---|---|---|
| **실제: 긍정** | TP (True Positive) | FN (False Negative, 2종 오류) |
| **실제: 부정** | FP (False Positive, 1종 오류) | TN (True Negative) |

### 실생활 예시로 보는 FP vs FN의 위험도

| 상황 | 더 위험한 오류 | 이유 |
|---|---|---|
| 화재 경보기 | FN (놓침) | 실제 화재를 못 잡으면 인명 피해로 직결 |
| 질병 진단 | FN (놓침) | 환자를 정상으로 오진하면 치료 시기를 놓침 |
| 스팸 필터 | FP (오탐) | 중요한 정상 메일을 스팸으로 분류하면 손실 |

**핵심 통찰:** 어떤 오류(FP vs FN)가 더 치명적인지는 문제(도메인)에 따라 다르다.

### 다중 클래스 혼동행렬

MNIST의 10×10 표를 바로 보기 전에, 원리를 파악하기 쉬운 작은 예시부터 살펴보자. 3개 클래스(개·고양이·토끼, 각 50마리씩 150마리) 예측 결과 예시:

![다중 클래스 혼동행렬 예시](/assets/img/posts/mnist-images/03_confusion_matrix.png)

- 대각선 값(45, 40, 40)은 올바른 예측
- 총 정확도 = (45+40+40)/150 = **83.3%**
- 가장 많은 오류: 고양이 → 토끼 오분류 (8건) → 고양이-토끼 구별 특징(털 패턴, 귀 모양) 추가 학습 필요

### MNIST 모델에 적용하기

같은 원리를 MNIST에 적용하면, 행과 열이 각각 0~9 숫자에 대응하는 10×10 혼동행렬이 만들어진다. 앞서 학습시킨 모델(`model`)과 검증 데이터(`test_loader`)로 직접 계산해볼 수 있다.

```python
from sklearn.metrics import confusion_matrix

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print(cm.shape)  # (10, 10)
```

`cm`의 대각선 값(`cm[i][i]`)은 숫자 `i`를 정확히 맞춘 개수이고, 대각선 밖의 값(`cm[i][j]`)은 실제로는 `i`인데 `j`로 잘못 예측한 개수다. 손글씨 숫자 인식에서는 모양이 비슷한 숫자쌍(예: 4와 9, 3과 5, 7과 1)에서 오분류가 몰리는 경향이 있다 — 앞의 개·고양이·토끼 예시에서 '털 패턴'이 헷갈리는 클래스 쌍을 찾아냈던 것과 같은 방식으로, 이 표를 통해 MNIST 모델이 어떤 숫자쌍을 가장 자주 헷갈리는지 바로 확인할 수 있다.

## 10. Precision과 Recall, 그리고 F1-Score {#10-precision-recall-f1}

### 정밀도(Precision)

$$Precision = \frac{TP}{TP + FP}$$

"양성이라고 한 예측이 얼마나 정확한가?" — 명중률에 비유. 스팸 필터(정상 메일 보호), 제품 추천 등에서 중요.

### 재현율(Recall)

$$Recall = \frac{TP}{TP + FN}$$

"실제 양성을 얼마나 찾아냈는가?" — 완전성에 비유. 암 진단, 사기 탐지 등 놓치면 안 되는 상황에서 중요.

### 임계값(Threshold)에 따른 트레이드오프

![임계값에 따른 정밀도-재현율 트레이드오프](/assets/img/posts/mnist-images/05_precision_recall_tradeoff.png)

| 임계값 | 특징 | Precision | Recall |
|---|---|---|---|
| 0.9 (엄격) | 확실한 것만 긍정 판정 | ↑ 높음 | ↓ 많이 놓침 |
| 0.5 (균형) | 표준 기준 | 중간 | 중간 |
| 0.3 (관대) | 의심스러우면 긍정 판정 | ↓ 오탐 많음 | ↑ 거의 안 놓침 |

**실무 팁:** 의료 진단처럼 놓치면 안 되는 경우는 낮은 임계값(높은 Recall), 스팸 필터처럼 오탐을 줄여야 하는 경우는 높은 임계값(높은 Precision)을 선택한다.

### F1-Score: 조화평균으로 균형 평가하기

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

일반 산술평균은 불균형을 숨기는 문제가 있다. 예를 들어 Precision=1.0, Recall=0.1이면:

- 산술평균 = 0.55 (중간처럼 보이는 착시)
- **F1(조화평균) = 0.18** (실제로 심각한 문제를 정직하게 반영)

F1은 Precision과 Recall 중 하나라도 낮으면 함께 낮아지므로, 두 지표 모두 높은 수준에서 균형을 이뤄야 높은 점수를 받을 수 있다.

## 11. 다중 클래스 평가: 평균 방식 비교 {#11-multiclass-averaging}

클래스가 여러 개일 때 클래스별 지표를 하나로 종합하는 3가지 방식:

| 방식 | 계산 | 특징 | 사용 시기 |
|---|---|---|---|
| **매크로(Macro)** | 클래스별 지표 단순 평균 | 모든 클래스를 동등하게 취급, 소수 클래스도 반영 | 소수 클래스 모니터링이 중요할 때 |
| **마이크로(Micro)** | 전체 TP/FP/FN 합산 후 계산 | 다수 클래스 영향이 큼, 전체 정확도와 유사 | 전체 샘플 기준 정확도가 중요할 때 |
| **가중(Weighted)** | 클래스별 샘플 수로 가중 평균 | 클래스 크기를 고려한 균형 잡힌 평가 | 현실적 성능 반영이 필요할 때 (실무에서 가장 무난) |

**예시:** 클래스 A(100개, F1=0.9), B(50개, F1=0.8), C(10개, F1=0.3)
- Macro-F1 = (0.9+0.8+0.3)/3 = **0.67**

## 12. ROC Curve와 AUC {#12-roc-auc}

ROC(Receiver Operating Characteristic) 곡선은 임계값을 연속적으로 바꿔가며 모델 성능을 시각화한 곡선이다.

- **X축**: FPR = FP / (FP+TN) — 낮을수록 좋음
- **Y축**: TPR = Recall — 높을수록 좋음

![ROC 곡선과 AUC](/assets/img/posts/mnist-images/04_roc_curve.png)

곡선이 왼쪽 위 모서리(FPR=0, TPR=1)에 가깝게 붙어있을수록 좋은 모델이다.

### AUC (Area Under Curve)

ROC 곡선 아래 면적으로, 모델 성능을 하나의 숫자(0~1)로 요약한다.

| AUC 값 | 의미 |
|---|---|
| 1.0 | 완벽한 분류기 |
| 0.9 ~ 1.0 | 매우 우수 |
| 0.8 ~ 0.9 | 우수 |
| 0.7 ~ 0.8 | 괜찮음 |
| 0.5 | 동전 던지기 수준 (무작위) |

> AUC = 0.9의 의미: 무작위로 선택한 긍정 샘플과 부정 샘플에서, 모델이 긍정 샘플에 더 높은 점수를 줄 확률이 90%.

**Precision/Recall/F1과의 차이점:** 이들은 특정 임계값 하나를 정해놓고 계산하지만, AUC는 임계값과 무관하게 모델 자체의 판별 능력을 평가한다.

## 13. 확률 보정(Calibration) {#13-calibration}

모델의 예측 확률이 실제 확률과 일치하도록 조정하는 과정. 의료 진단, 자율주행, 금융 리스크 평가처럼 확률 자체를 의사결정에 사용하는 경우 필수적이다.

### 문제 상황: 과신(Over-confident)

모델이 "0.8 확률"이라고 예측한 샘플 100개를 모았을 때:

- **이상적:** 80개가 실제 긍정
- **현실(과신):** 60개만 긍정

기상 예보에 비유하면, "비 올 확률 80%"라고 말했다면 10번 중 8번은 실제로 비가 와야 한다. 6번만 온다면 보정이 필요하다.

### 해결책: 온도 스케일링(Temperature Scaling)

Softmax에 온도(T) 파라미터를 추가해 확률 분포를 조정한다.

| T 값 | 효과 |
|---|---|
| T = 1 | 원래 확률 (변화 없음) |
| T > 1 | 확률 분포가 평평해짐 (덜 확신) — 과신 문제 보정 |
| T < 1 | 확률 분포가 날카로워짐 (더 확신) — 과소신 문제 보정 |

**장점:** 단일 파라미터만 조정하면 되고, **모델 재학습이 불필요**하다.

## 14. 클래스 불균형(Class Imbalance) 해결 전략 {#14-class-imbalance}

### 문제 상황

정상 메일 990개, 스팸 10개인 경우, 모든 메일을 "정상"으로 예측해도 **99% 정확도**를 얻지만 스팸을 단 하나도 걸러내지 못하는 쓸모없는 모델이 된다. 이것이 "정확도만으로 모델을 평가하면 안 되는" 대표적인 함정이다.

### 4가지 해결 전략

| 전략 | 원리 | 장점 | 사용 시기 |
|---|---|---|---|
| **가중 손실(Weighted Loss)** | 소수 클래스 손실에 큰 가중치 부여 | 구현 간단, 데이터 변경 불필요 | 기본적인 첫 시도 |
| **SMOTE** | 기존 샘플 사이를 보간해 새 소수 클래스 샘플 생성 | 과적합 감소, 일반화 성능 향상 | 표 데이터 |
| **데이터 증강** | 회전·반전·노이즈 추가로 소수 클래스 증강 | 실제로 새로운 정보 생성 | 이미지·텍스트 데이터 |
| **앙상블 기법** | 다수 클래스에서 서브셋 샘플링 후 여러 모델 학습·결합 | 정보 손실 최소화, 안정적 성능 | 최종 프로덕션 모델 |

---

## 15. 전체 정리 {#15-summary}

이 수업은 크게 두 파트로 나눌 수 있다.

```
[1부] 딥러닝 기초
문제 정의(MNIST) → 신경망 구조(은닉층) → ReLU → GPU/미니배치
→ 데이터 파이프라인(Dataset/Transforms/DataLoader) → 모델 정의(nn.Module)
→ 학습 결과 → 경사소실/람다/배치사이즈 심화

[2부] 모델 평가 심화
혼동행렬(TP/TN/FP/FN) → 다중클래스 확장 → Precision/Recall/F1
→ ROC/AUC → 확률보정(Calibration) → 클래스 불균형 해결
```

### 핵심 메시지 3가지

1. **"정확도 하나의 숫자"만 믿지 말 것.** 혼동행렬 기반의 다양한 지표로 모델을 다각도로 평가해야 한다.
2. **상황(도메인)에 따라 우선순위가 다르다.** 의료 진단은 Recall, 스팸 필터는 Precision이 더 중요할 수 있다.
3. **분류 성능뿐 아니라 확률의 신뢰도와 데이터 균형도 함께 고려해야** 실무에서 쓸 수 있는 모델이 된다.

---

*이 글은 PyTorch 기반 MNIST 다중분류 수업 자료를 정리한 것입니다. 코드 예제는 실제 동작을 위해 데이터 경로, 모델 하이퍼파라미터 등을 환경에 맞게 조정해야 할 수 있습니다.*
