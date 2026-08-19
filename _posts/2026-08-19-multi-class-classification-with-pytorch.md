---
title: "PyTorch로 이해하는 다중 분류(Multi-class Classification) 완전 정리"
date: 2026-08-19 09:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, multi-class-classification, softmax, cross-entropy-loss, machine-learning]
description: "붓꽃 3품종 분류 예제로 이진 분류와 다중 분류의 차이, 소프트맥스와 교차 엔트로피의 동작 원리, nn.CrossEntropyLoss의 표준 설계를 처음부터 끝까지 정리했다."
---

지금까지 우리가 다뤄온 분류 문제는 대부분 "둘 중 하나"를 고르는 **이진 분류(Binary Classification)**였습니다. 이번 글에서는 한 걸음 더 나아가, 셋 이상의 선택지 중 하나를 고르는 **다중 분류(Multi-class Classification)**를 처음부터 끝까지 정리해봅니다.

붓꽃(iris) 품종을 `setosa`, `versicolor`, `virginica` 세 가지로 분류하는 예제를 중심으로, 개념 → 수식 → PyTorch 코드까지 순서대로 따라가 보겠습니다.

## 📌 목차

1. 이진 분류 vs 다중 분류: 무엇이 달라지는가
2. 여러 개의 분류기가 동시에 작동한다
3. 소프트맥스(Softmax): 점수를 확률로 바꾸는 함수
4. 교차 엔트로피(Cross Entropy): 예측과 정답의 차이 측정
5. 파이토치의 표준 설계: `nn.CrossEntropyLoss`
6. 전체 계산 흐름 한눈에 보기
7. 실습: iris 데이터셋으로 다중 분류 모델 만들기
8. 참고: 같은 결과를 내는 3가지 구현 패턴
9. 마치며: 핵심 요약

---

## 1. 이진 분류 vs 다중 분류: 무엇이 달라지는가

이진 분류가 동전의 앞/뒤를 맞추는 게임이었다면, 다중 분류는 여러 개의 문 중에 진짜 보물이 숨겨진 단 하나의 문을 찾는 게임과 같습니다. 선택지가 많아진 만큼, 조금 더 정교한 전략이 필요합니다.

붓꽃 품종을 세 가지로 분류하려면, 모델은 "이 붓꽃이 setosa일 확률은 30%, versicolor일 확률은 60%, virginica일 확률은 10%입니다"처럼 **각 항목에 대한 확률을 모두 제시**해야 합니다. 이진 분류에서는 "합격일 확률 70%" 하나만 내놓으면 나머지(불합격 확률 30%)가 자동으로 계산됐지만, 다중 분류에서는 그렇게 되지 않습니다.

이 차이는 결국 **모델 출력의 차원**으로 정리됩니다.

| 구분 | 이진 분류 | 다중 분류 |
|---|---|---|
| 출력 | 정답일 확률 단 하나 | 각 클래스별 확률을 모두 계산 |
| 출력 차원 | 1차원 | N차원 (N = 클래스 개수) |
| 파라미터 | 가중치 **벡터** | 가중치 **행렬** |

![이진 분류와 다중 분류의 구조 비교](/assets/img/posts/multi-class-classification/01_binary_vs_multiclass.png)

가장 큰 변화는 출력의 차원입니다. 이진 분류 모델의 최종 출력은 1차원이었지만, N개의 그룹으로 분류하는 다중 분류 모델의 출력은 N차원이 됩니다.

---

## 2. 여러 개의 분류기가 동시에 작동한다

다중 분류를 직관적으로 이해하는 가장 좋은 방법은, **모델 내부에 N개의 작은 분류기가 있고 각 분류기가 특정 그룹을 전담한다**고 생각하는 것입니다.

- 분류기 0 → "이 데이터가 setosa(0번 그룹)에 속할 확률은?"
- 분류기 1 → "이 데이터가 versicolor(1번 그룹)에 속할 확률은?"
- 분류기 2 → "이 데이터가 virginica(2번 그룹)에 속할 확률은?"

모델은 입력 데이터에 대해 **모든 분류기의 확률 값을 동시에 계산**하고, 이 중 **가장 높은 확률을 제시하는 분류기의 그룹을 최종 예측값**으로 선택합니다.

![여러 개의 분류기가 동시에 작동하는 구조](/assets/img/posts/multi-class-classification/02_multiple_classifiers.png)

### 가중치 벡터에서 가중치 행렬로

3개의 분류기는 각자 자신만의 가중치 벡터를 하나씩 가지고 있습니다. 이 벡터들을 하나로 쌓아 놓은 것이 바로 **가중치 행렬(Weight Matrix)**입니다.

요리에 비유하면 이렇게 정리할 수 있습니다.

| 구분 | 이진 분류 | 다중 분류 |
|---|---|---|
| 가중치 | 가중치 벡터 사용 | 가중치 행렬 사용 |
| 비유 | 하나의 레시피 | 여러 레시피가 담긴 요리책 |
| 계산 | 입력 벡터와 가중치 벡터의 **내적**으로 출력 1개 계산 | 입력 벡터와 가중치 행렬의 **곱셈**으로 여러 출력 동시 계산 |

동일한 재료(입력)를 가지고 각기 다른 레시피(가중치 행)를 적용하여 여러 맛의 요리(출력)를 각각 계산하는 셈입니다. 덕분에 입력 벡터 하나를 가중치 행렬에 곱하기만 하면, 분류기 0, 1, 2의 출력값이 **한 번의 행렬 연산으로 동시에** 계산됩니다.

```python
# 이진 분류: 가중치 벡터(1차원 출력)
# W.shape = (input_dim,)
score = x @ W + b          # 스칼라 1개

# 다중 분류: 가중치 행렬(N차원 출력)
# W.shape = (input_dim, num_classes)
scores = x @ W + b         # 클래스 개수(N)만큼의 벡터
```

---

## 3. 소프트맥스(Softmax): 점수를 확률로 바꾸는 함수

가중치 행렬로 계산한 값(원시 점수, logit)은 아직 확률이 아닙니다. 음수가 나올 수도 있고, 값을 다 더해도 1이 되지 않습니다. 이걸 우리가 이해할 수 있는 "확률"로 바꿔주는 것이 바로 **소프트맥스 함수**입니다.

> 이진 분류에서 시그모이드 함수가 출력을 0과 1 사이의 확률로 바꿔주었다면, 다중 분류에서는 소프트맥스 함수가 그 역할을 합니다.

소프트맥스는 세 단계를 거칩니다.

1. **N개의 출력값** — 모델의 원시 출력(logit)
2. **확률 분포 변환** — 각 값의 상대적 비중을 계산
3. **합이 1인 확률** — 최종적으로 모든 확률의 합이 정확히 1이 됨

![소프트맥스 함수로 점수를 확률로 변환하는 과정](/assets/img/posts/multi-class-classification/03_softmax_transform.png)

이름이 '소프트맥스'인 이유는, 가장 큰 값만 1로 만드는 극단적인 방식(hard max)이 아니라, 가장 큰 값의 확률이 가장 크도록 하면서도 다른 값들에게도 어느 정도의 확률을 부드럽게(soft) 할당해주기 때문입니다.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.1, 0.8, -1.3])
probs = F.softmax(logits, dim=0)
print(probs)
# tensor([0.7658, 0.2087, 0.0256])  ← 합계는 항상 1
```

---

## 4. 교차 엔트로피(Cross Entropy): 예측과 정답의 차이 측정

소프트맥스로 확률을 구했다면, 이제 이 예측이 실제 정답과 얼마나 다른지 측정할 손실 함수가 필요합니다. 다중 분류에서도 교차 엔트로피를 사용하지만, 계산 방식에 차이가 있습니다.

계산은 3단계로 이루어집니다.

1. **로그 적용** — 소프트맥스 출력(각 클래스일 확률)에 로그를 취함
2. **정답 요소 추출** — 여러 로그 값 중 실제 정답에 해당하는 클래스의 값만 추출
3. **손실 계산** — 이 값에 음수를 붙여 손실로 사용 (확률이 1에 가까울수록 손실은 작아짐)

정답 클래스의 예측 확률이 1에 가까우면(잘 맞췄으면) 손실은 0에 가까워지고, 0에 가까우면(완전히 틀렸으면) 손실은 매우 커집니다. 즉, **모델이 정답을 자신 있게 맞출수록 벌점이 작아지는 구조**입니다.

### NLLLoss: 정답 요소를 추출하는 역할

이 계산의 마지막 단계, 즉 "정답 클래스에 해당하는 값만 골라 음수를 붙이는" 역할을 실제로 수행하는 함수가 **NLLLoss(Negative Log Likelihood Loss)**입니다.

3개의 샘플이 있고 각각의 정답 레이블이 `[0, 1, 2]`라고 해봅시다. 모델이 각 샘플마다 **자신의 정답 클래스**에 해당하는 로그 확률만 뽑아 `[log_prob_sample0, log_prob_sample1, log_prob_sample2]`를 얻었다면, NLLLoss는 다음을 계산합니다.

```
loss = -(log_prob_sample0 + log_prob_sample1 + log_prob_sample2) / 3
```

파이토치는 정답을 원-핫 인코딩으로 바꾸지 않고, **정수형 레이블 `[0, 1, 2]`를 그대로 인덱스처럼 사용**합니다. 그래서 정답 레이블 텐서는 반드시 **long(정수) 타입**이어야 합니다.

```python
import torch.nn as nn

y = torch.tensor([0, 1, 2], dtype=torch.long)   # 정답 레이블은 반드시 long 타입!
```

---

## 5. 파이토치의 표준 설계: `nn.CrossEntropyLoss`

이론적으로는 `선형 함수 → 소프트맥스 → 로그 → 정답 요소 추출`이라는 4단계를 거쳐야 하지만, 파이토치는 이 전체 과정을 `nn.CrossEntropyLoss` 하나로 압축해서 제공합니다.

`nn.CrossEntropyLoss`는 내부적으로 **소프트맥스 + 로그 + 정답 요소 추출(NLLLoss)**을 모두 포함하고 있어 수치적으로 안정적이고 효율적입니다.

![파이토치의 모델-손실함수 역할 분담 구조](/assets/img/posts/multi-class-classification/05_pytorch_design.png)

이 설계에서 가장 중요한 규칙은 다음과 같습니다.

> **예측 모델(net)은 활성화 함수 없이 `nn.Linear`의 출력(raw score, logit)을 그대로 반환해야 합니다.**

```python
import torch.nn as nn

class IrisClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)   # 활성화 함수(소프트맥스)를 넣지 않는다!

net = IrisClassifier(input_dim=2, num_classes=3)
criterion = nn.CrossEntropyLoss()   # 소프트맥스 + 로그 + NLLLoss를 내부에서 처리
```

### ⚠️ 자주 하는 실수: 소프트맥스 중복 적용

`nn.CrossEntropyLoss`가 이미 소프트맥스를 내부적으로 계산하기 때문에, 모델(net) 안에 소프트맥스를 직접 넣으면 **소프트맥스를 두 번 적용하는 실수**가 됩니다. 모델은 항상 raw score만 반환하도록 만드세요.

### 왜 이렇게 나눠서 처리할까?

소프트맥스 계산에는 지수함수(exp)가 들어가기 때문에, 숫자가 너무 크거나 작으면 오버플로우나 계산 오차가 발생할 수 있습니다. `CrossEntropyLoss`는 소프트맥스와 로그 계산을 수학적으로 합쳐서 처리하는 안정적인 방식을 내부적으로 사용하기 때문에, 따로따로 계산하는 것보다 훨씬 안전합니다.

---

## 6. 전체 계산 흐름 한눈에 보기

지금까지 다룬 내용을 하나의 파이프라인으로 정리하면 다음과 같습니다.

![다중 분류 모델의 전체 계산 흐름](/assets/img/posts/multi-class-classification/04_full_pipeline.png)

```
입력 벡터(x)
  → 가중치 행렬(W) 곱셈
  → 원시 출력(logits, N개)
  → 소프트맥스(확률로 변환)
  → 교차 엔트로피(정답과 비교)
  → 손실(loss) 계산 완료
```

---

## 7. 실습: iris 데이터셋으로 다중 분류 모델 만들기

### 7-1. 데이터 준비

iris 데이터셋은 총 150개의 샘플로 구성되어 있습니다. 이를 **훈련용 75개 / 검증용 75개**로 나눕니다.

과적합(overfitting)을 방지하기 위해 데이터를 나누는 것이 중요합니다. 모든 데이터로 학습하면 모델이 데이터를 단순히 "외워버릴" 수 있기 때문입니다. 학습에 사용하지 않은 검증 데이터로 모델의 진짜 실력을 확인합니다.

시각화를 위해 원래 4개인 특징(꽃받침 길이·너비, 꽃잎 길이·너비) 중 **꽃받침 길이와 꽃잎 길이 2개만** 선택해 산점도로 데이터 분포를 미리 확인합니다.

```python
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, [0, 2]]   # 꽃받침 길이, 꽃잎 길이만 선택 (시각화 목적)
y = iris.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# 텐서 변환: 입력은 float, 정답 레이블은 반드시 long
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
```

### 7-2. 모델 정의

**입력 2개, 출력 3개**를 가진 선형 모델을 정의합니다. 가중치는 벡터가 아닌 (2, 3) 크기의 행렬 형태가 됩니다.

```python
import torch.nn as nn

model = nn.Linear(in_features=2, out_features=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

### 7-3. 학습 (경사 하강법)

경사 하강법으로 모델을 학습시키면서, 매 에포크마다 손실(loss)과 정확도(accuracy)를 기록합니다.

```python
train_losses, val_losses = [], []

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)          # (75, 3) 크기의 raw score
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
```

### 7-4. 결과 확인: `torch.max`로 예측 클래스 뽑아내기

모델은 (배치 크기, 클래스 수) 형태의 텐서를 출력합니다. 75개의 검증 데이터에 대해 3개 클래스를 예측하면 **(75, 3) 텐서**가 나옵니다.

여기서 실제로 필요한 것은 "어느 클래스로 예측했는가"이므로, `torch.max(outputs, 1)`을 사용해 **행(row)마다 가장 높은 값의 인덱스**를 뽑아냅니다.

```python
with torch.no_grad():
    outputs = model(X_val)
    predicted = torch.max(outputs, 1)[1]   # [1]로 indices만 추출
    accuracy = (predicted == y_val).float().mean()
    print(f"검증 정확도: {accuracy.item():.2%}")
```

`torch.max(outputs, 1)`은 `values`(최댓값 자체)와 `indices`(최댓값의 위치)를 함께 반환하는데, 우리가 필요한 것은 `indices`입니다. 이 인덱스가 바로 모델이 예측한 클래스 레이블(0 = setosa, 1 = versicolor, 2 = virginica)이 됩니다.

### 7-5. 모델 출력값을 사람이 보기 좋게 확인하기

`nn.CrossEntropyLoss`를 사용했기 때문에, 학습된 모델은 소프트맥스를 거치지 않은 원시 출력(logits)을 그대로 반환합니다. `[2.1, 0.8, -1.3]`처럼 음수가 섞여 있거나 합이 1이 아닌 값이 나오는 것은 정상입니다.

모델의 실제 예측 확신도를 사람이 보기 좋은 형태로 확인하고 싶다면, 출력값에 **직접 소프트맥스를 적용**해야 합니다.

```python
with torch.no_grad():
    sample_output = model(X_val[:1])
    print("원시 출력(logits):", sample_output)
    print("확률(softmax 적용 후):", F.softmax(sample_output, dim=1))
```

### 7-6. 학습 곡선으로 성능 모니터링하기

학습 곡선은 에포크가 진행됨에 따라 손실과 정확도가 어떻게 변하는지를 보여주는 중요한 지표입니다.

- **손실 곡선**: 이상적으로는 에포크가 진행될수록 부드럽게 감소해 낮은 값으로 수렴
- **정확도 곡선**: 손실과 반대로, 부드럽게 증가해 높은 값으로 수렴
- **안정성 확인**: 훈련 손실과 검증 손실이 비슷하게 움직이면 학습이 안정적으로 이루어지고 있다는 뜻 (두 곡선이 벌어지기 시작하면 과적합 신호)

```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

---

## 8. 참고: 같은 결과를 내는 3가지 구현 패턴

표준 패턴(`nn.Linear` + `CrossEntropyLoss`) 외에도 이론적으로 동일한 결과를 내는 다른 구현 방법들이 있습니다.

| 패턴 | 구성 | 평가 |
|---|---|---|
| **패턴 1 (표준)** | 선형 모델 + `CrossEntropyLoss` | ⭐ 가장 안정적이고 권장되는 방식 |
| 패턴 2 | `LogSoftmax` + `NLLLoss` | 표준 패턴과 동일한 결과 |
| 패턴 3 | `Softmax` + 수동 log + `NLLLoss` | 번거롭고 수치적으로 불안정, 비권장 |

패턴 2는 모델이 `LogSoftmax`까지 계산하여 로그 확률을 출력하고, 손실 함수는 `NLLLoss`로 마지막 '추출' 작업만 담당합니다. 패턴 3처럼 소프트맥스와 로그를 따로따로 계산하면 지수함수 계산 과정에서 오차가 누적될 위험이 있으므로, 특별한 이유가 없다면 **패턴 1(표준 방식)을 사용하는 것이 안전**합니다.

```python
# 패턴 2 예시: LogSoftmax + NLLLoss (표준 패턴과 수학적으로 동일)
class IrisClassifierV2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return self.log_softmax(x)

model_v2 = IrisClassifierV2(2, 3)
criterion_v2 = nn.NLLLoss()   # 정답 요소 추출만 담당
```

---

## 9. 마치며: 핵심 요약

이번 글에서 다룬 내용을 3가지로 요약하면 다음과 같습니다.

1. **출력 차원의 확장** — 1개 출력에서 N개 출력으로, 가중치 벡터에서 가중치 행렬로 변화
2. **소프트맥스의 역할** — N개의 출력값을 확률 분포로 변환, 모든 확률의 합이 1
3. **파이토치의 효율적 설계** — `CrossEntropyLoss`가 소프트맥스부터 손실 계산까지 한 번에 처리

다중 분류를 이해하면 손글씨 숫자 이미지 분류, 뉴스 기사 카테고리 분류 등 훨씬 다채롭고 현실적인 문제들을 해결할 수 있습니다. 다음 글에서는 iris 데이터의 4개 특징을 모두 사용하는 입력 확장 실습을 다뤄보겠습니다.
