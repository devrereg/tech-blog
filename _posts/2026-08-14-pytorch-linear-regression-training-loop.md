---
title: "파이토치로 이해하는 선형 회귀 모델 아키텍처와 학습 루프(Training Loop) 정리"
date: 2026-08-14 00:10:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, linear-regression]
description: "nn.Module로 커스텀 선형 회귀 모델을 정의하고, zero_grad → forward → loss → backward → step으로 이어지는 PyTorch 학습 루프의 5단계를 코드와 함께 정리했다."
---

## 📌 목차
1. PyTorch 모델 빌드의 핵심: `__init__`과 `forward`
2. 가중치(Weight)와 편향(Bias) 초기화
3. `nn.Module`을 활용한 커스텀 회귀 모델 설계
4. 손실 함수(Loss)와 옵티마이저(Optimizer) 구성
5. 파이토치 학습 루프(Training Loop)의 5단계 메커니즘

---

## 1. PyTorch 모델 빌드의 핵심: `__init__`과 `forward`

PyTorch에서 `nn.Module`을 상속받아 커스텀 모델 클래스를 정의할 때, 가장 기본이 되는 두 메서드의 역할은 다음과 같다.

* **`__init__(self, ...)`**: 모델에 필요한 레이어(층) 및 학습 가능한 파라미터들을 정의하고 초기화한다.
* **`forward(self, x)`**: 입력 데이터 `x`를 받아 `__init__`에서 정의한 레이어들을 순서대로 통과시키며, 순전파(Forward) 연산 흐름을 진행하고 결과를 반환한다.

---

## 2. 가중치(Weight)와 편향(Bias) 초기화

모델의 초기 학습 상태를 제어하기 위해 가중치와 편향을 특정 값으로 고정해야 할 때가 있다. PyTorch에서는 인플레이스(In-place) 연산 함수인 `nn.init.constant_` 등을 활용해 이를 명시적으로 제어할 수 있다.

```python
import torch
import torch.nn as nn

# 입력 특성 3개, 출력 특성 1개인 선형 레이어 생성
l = nn.Linear(in_features=3, out_features=1)

# 가중치는 2.0으로, 편향은 0.5로 초기화 (y = 2*x1 + 2*x2 + 2*x3 + 0.5)
nn.init.constant_(l.weight, 2.0)
nn.init.constant_(l.bias, 0.5)

print(f"l.weight: {l.weight.data}")  # 크기: (1, 3)
print(f"l.bias: {l.bias.data}")      # 크기: (1,)
```

---

## 3. `nn.Module`을 활용한 커스텀 회귀 모델 설계

선형 회귀 모델을 직접 클래스로 구현할 때는 부모 클래스인 `nn.Module`의 초기화 함수를 먼저 호출한 뒤, 필요한 선형 레이어(`nn.Linear`)를 등록한다.

```python
class MyRegressionModel(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()  # 부모 클래스의 __init__ 필수 호출
        self.linear_layer = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear_layer(x)

# 특성 5개를 받아 1개를 출력하는 모델 인스턴스 생성
net = MyRegressionModel(n_input=5, n_output=1)
```

---

## 4. 손실 함수(Loss)와 옵티마이저(Optimizer) 구성

모델을 학습시키기 위해서는 예측값과 정답 간의 오차를 계산할 **손실 함수**와, 이 오차를 바탕으로 가중치를 업데이트할 **옵티마이저**가 필요하다. 회귀 문제에서는 주로 MSE(평균 제곱 오차)와 SGD(확률적 경사 하강법) 조합이 기본으로 사용된다.

```python
import torch.optim as optim

learning_rate = 0.01

# 1. 회귀 문제용 평균 제곱 오차(MSE) 손실 함수 정의
criterion = nn.MSELoss()

# 2. 모델 파라미터와 학습률을 전달하여 SGD 옵티마이저 정의
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
```

---

## 5. 파이토치 학습 루프(Training Loop)의 5단계 메커니즘

PyTorch에서 가중치를 업데이트하는 학습 반복 루프는 항상 아래의 5단계 표준 흐름(Best Practice)을 따른다. 경사가 누적되는 파이토치의 특성상, **새로운 연산을 시작하기 전에 반드시 경사를 초기화**해 주는 것이 핵심이다.

```python
# 가상의 데이터셋 생성 (배치 크기 10, 특성 5)
inputs = torch.randn(10, 5)
labels = torch.randn(10, 1)

# --- 학습 루프 1회 반복 흐름 ---

# Step 1. 경사 초기화
# 이전 반복(Iteration)에서 계산되어 텐서에 남아있는 미분값(gradient)을 0으로 비운다.
optimizer.zero_grad()

# Step 2. 예측 계산 (Forward Pass)
# 모델에 입력 데이터를 통과시켜 예측값(outputs)을 도출한다.
outputs = net(inputs)

# Step 3. 손실 계산
# 손실 함수를 통해 예측값과 실제 정답(labels) 사이의 오차를 계산한다.
loss = criterion(outputs, labels)

# Step 4. 경사 계산 (Backward Pass / 역전파)
# loss를 시작으로 연산 그래프를 역추적하여 각 파라미터에 대한 미분(grad)을 계산한다.
loss.backward()

# Step 5. 파라미터 수정 (Optimization Step)
# 계산된 미분값과 설정된 학습률(lr)을 바탕으로 모델의 가중치와 편향을 업데이트한다.
optimizer.step()

print(f"Calculated Loss: {loss.item():.4f}")
```

---

## 💡 핵심 요약 및 회고

* `scikit-learn`과 달리 `PyTorch`는 모델 아키텍처, 손실 함수, 가중치 업데이트가 독립적인 컴포넌트로 분리되어 있어 자유도가 높다.
* 학습 루프의 핵심 주기인 **`Zero-Grad → Forward → Loss → Backward → Step`** 구조를 명확히 이해해야 추후 복잡한 딥러닝 아키텍처로 확장할 때 디버깅이 수월해진다.
