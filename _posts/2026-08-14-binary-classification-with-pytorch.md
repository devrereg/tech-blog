---
title: "이진 분류(Binary Classification) 완전 정복: 붓꽃 데이터로 배우는 로지스틱 회귀"
date: 2026-08-14 09:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, binary-classification, logistic-regression, sigmoid, cross-entropy]
math: true
description: "붓꽃 데이터셋으로 이진 로지스틱 회귀 모델을 처음부터 끝까지 구현하며 시그모이드, 교차 엔트로피, 정확도, 과학습 개념을 정리했다."
---

## 📌 목차
1. 이진 분류란?
2. 데이터셋 준비 (붓꽃 예시)
3. 이진 로지스틱 회귀 모델 구조
4. 정확도(Accuracy)
5. 훈련 데이터 vs 검증 데이터
6. 시그모이드 함수
7. 교차 엔트로피 함수(손실 함수)
8. 학습 과정 (경사 하강법 5단계)
9. `BCELoss` vs `BCEWithLogitsLoss`
10. 전체 실습 코드

---

## 1. 이진 분류란?

회귀(Regression)가 "값 맞히기"(연속적인 숫자 예측)라면, **이진 분류(Binary Classification)**는 "편 가르기"(두 그룹 중 하나로 분류)다.

* 스팸 메일인가, 아닌가?
* 사진 속 동물이 고양이인가, 강아지인가?
* 이 거래가 정상인가, 사기인가?

이렇게 정답이 딱 두 가지 범주로 나뉘는 문제를 다루는 것이 이진 분류다.

---

## 2. 데이터셋 준비 (붓꽃 예시)

실습에는 사이킷런에 내장된 붓꽃(iris) 데이터셋을 사용한다.

* 원본: 150개 데이터, 4개 특성(꽃받침 길이/폭, 꽃잎 길이/폭), 3개 품종
* 이진 분류 실습용으로는 이 중 **2개 품종(100개)** × **2개 특성**(꽃받침 길이/폭)만 추출해서 쓴다.

품종이 3개인 원본을 그대로 쓰면 다중 분류가 되므로, 문제를 단순화하기 위해 품종 2개만 남기고 시각화하기 쉽도록 특성도 2개로 줄인다.

---

## 3. 이진 로지스틱 회귀 모델 구조

```
입력[2] → nn.Linear(가중치·편향) → nn.Sigmoid → 출력[1](확률)
```

앞서 다룬 회귀 모델과 구조는 거의 같지만, 마지막에 **시그모이드(Sigmoid)**가 하나 더 붙는다는 점이 가장 큰 차이다. 선형 계층이 뱉어낸 제한 없는 숫자를 시그모이드가 0~1 사이의 확률로 눌러주는 것이다.

---

## 4. 정확도(Accuracy)

$$정확도 = \frac{정답\ 건수}{전체\ 건수}$$

회귀 모델의 성능은 손실 값만 봐서는 감이 잘 안 오지만, 분류 모델은 "맞았다/틀렸다"로 평가하기 때문에 훨씬 직관적이다. 참고로 아무것도 학습하지 않은 초기(무작위) 상태의 이진 분류 정확도는 반반 찍기 수준인 **50%** 근처에서 시작한다.

---

## 5. 훈련 데이터 vs 검증 데이터

| 구분 | 역할 | 비유 |
| --- | --- | --- |
| 훈련 데이터 | 모델 학습에 사용 | 연습 문제 |
| 검증 데이터 | 성능 평가 전용 (학습에 사용 X) | 모의고사 |

모델이 훈련 데이터만 잘 맞히고 처음 보는 데이터는 못 맞히는 현상을 **과학습(Overfitting)**이라고 부른다. 훈련 정확도는 계속 오르는데 검증 정확도는 오르지 않거나 떨어진다면 과학습을 의심해야 한다.

---

## 6. 시그모이드 함수

$$f(x) = \frac{1}{1+e^{-x}}$$

* 입력이 무엇이든 출력은 **항상 0~1 사이**의 값이 된다.
* $x = 0$일 때 출력은 정확히 **0.5**다.
* 선형 함수가 만들어낸 "제한 없는 숫자"를 "확률"로 바꿔주는 역할을 한다.
* 예측 기준: 출력이 0.5보다 크면 1, 작으면 0으로 판단한다.

---

## 7. 교차 엔트로피 함수 (손실 함수)

회귀에서 MSE를 쓴다면, 분류에서는 **교차 엔트로피(Cross Entropy)**를 쓴다. 핵심은 **확신을 가지고 틀렸을 때 더 큰 페널티**를 준다는 점이다.

* "정답은 사과인 것 같아요" (틀림) → 벌점 작음
* "정답은 100% 바나나입니다!" (틀림) → 벌점 큼

애매하게 틀린 것보다 자신만만하게 틀린 것을 훨씬 더 강하게 혼내는 셈이다.

---

## 8. 학습 과정 (경사 하강법 5단계)

1. `optimizer.zero_grad()` — 경사값 초기화
2. 순전파 — 예측값 계산
3. 손실 계산 — 교차 엔트로피
4. `loss.backward()` — 역전파(경사 계산)
5. `optimizer.step()` — 파라미터 업데이트

> ⚠️ `zero_grad()`를 빼먹으면 이전 반복의 경사값이 계속 누적되어 잘못된 방향으로 파라미터가 업데이트된다.

이 5단계를 수천 번 반복하면 초기 50%였던 정확도가 점점 올라가고, 훈련·검증 손실/정확도 곡선이 비슷하게 움직인다면 과적합 없이 잘 일반화되고 있다는 신호다. 결국 로지스틱 회귀가 학습하는 것은 두 그룹을 나누는 **결정 경계(직선) 하나**이며, 이 직선의 위치와 기울기를 WEIGHT와 BIAS 값이 결정한다.

---

## 9. `BCELoss` vs `BCEWithLogitsLoss`

| 구분 | 기존 방식 (`BCELoss`) | 권장 방식 (`BCEWithLogitsLoss`) |
| --- | --- | --- |
| 모델 | Linear → Sigmoid | Linear만 |
| 손실 함수 | `nn.BCELoss()` | `nn.BCEWithLogitsLoss()` |
| 예측 기준 | 확률 0.5 | 로짓(logit) 0.0 |
| 특징 | 시그모이드와 손실 계산을 두 단계로 분리 | 시그모이드+손실 계산을 통합 → **수치적으로 더 안정적** |

두 방식의 차이가 왜 생기는지, `BCEWithLogitsLoss`가 왜 수치적으로 더 안전한지는 [이전 글]({% post_url 2026-08-13-bce-with-logits-vs-sigmoid %})에서 자세히 다뤘다.

---

## 10. 전체 실습 코드

아래 코드는 붓꽃 데이터셋으로 위 개념 전체를 처음부터 끝까지 구현한 것이다. 앞서 정리한 `BCEWithLogitsLoss` 방식을 사용했다.

```python
# ==========================================
# 이진 분류 (Binary Classification) 전체 실습 코드
# 붓꽃(iris) 데이터셋을 활용한 이진 로지스틱 회귀
# ==========================================

# ---------- 0. 필요한 라이브러리 불러오기 ----------
import numpy as np                                     # 수치 연산 라이브러리
import torch                                            # 파이토치 메인 라이브러리
import torch.nn as nn                                   # 신경망 계층(Linear 등)을 담은 모듈
import torch.optim as optim                             # 경사 하강법 등 최적화 알고리즘 모듈
from sklearn.datasets import load_iris                  # 사이킷런 내장 붓꽃 데이터셋 로더
from sklearn.model_selection import train_test_split    # 훈련/검증 데이터 분리 함수
import matplotlib.pyplot as plt                         # 손실·정확도 곡선 시각화용
import matplotlib

# 한글 폰트 설정 (matplotlib 기본 폰트는 한글을 지원하지 않아 그래프 글자가 깨질 수 있음)
# 운영체제에 맞는 한글 폰트가 설치되어 있어야 적용됨
# Windows: 'Malgun Gothic' / Mac: 'AppleGothic' / Linux: 나눔고딕 등 설치 필요
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False        # 그래프에서 마이너스(-) 기호 깨짐 방지


# ---------- 1. 데이터 불러오기 ----------
iris = load_iris()            # 붓꽃 데이터셋 로드 (150개 데이터, 4개 특성, 3개 품종)
X_all = iris.data             # 특성 데이터, shape: (150, 4) → 꽃받침 길이/폭, 꽃잎 길이/폭
y_all = iris.target           # 정답 레이블, shape: (150,) → 0:setosa, 1:versicolor, 2:virginica


# ---------- 2. 데이터 추출 (이진 분류를 위한 단순화) ----------
# 3개 품종 중 2개 품종(0: setosa, 1: versicolor)만 사용
# 데이터가 품종별로 순서대로 50개씩 저장되어 있어 레이블이 0 또는 1인 데이터만 필터링하면 100개가 됨
binary_mask = y_all < 2             # 레이블이 0 또는 1인 데이터만 True로 표시하는 마스크
X = X_all[binary_mask]              # 마스크 적용 → 100개 데이터만 추출
y = y_all[binary_mask]              # 정답도 동일하게 100개만 추출

# 시각화를 쉽게 하기 위해 4개 특성 중 앞의 2개(꽃받침 길이, 꽃받침 폭)만 사용
X = X[:, :2]                        # shape: (100, 2)


# ---------- 3. 훈련 데이터와 검증(테스트) 데이터 분리 ----------
# 전체 100개 데이터를 훈련 70개 + 검증 30개로 무작위 분할
# test_size=0.3 → 검증 데이터 비율 30%
# random_state를 고정하면 실행할 때마다 동일하게 분할되어 결과 재현 가능
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ---------- 4. 텐서 변환 ----------
# 넘파이 배열을 파이토치 텐서(Tensor)로 변환
# dtype=torch.float32 : 연산 속도와 정밀도의 균형을 위한 표준 실수형
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)

# 정답 텐서는 shape을 (N, 1) 형태로 맞춰야 함
# 모델의 출력 shape이 [배치 크기, 1]이므로, 정답 shape을 맞추지 않으면 손실 계산 시 오류 발생
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)


# ---------- 5. 모델 정의 ----------
class Net(nn.Module):
    """
    2입력 1출력 이진 로지스틱 회귀 모델

    BCEWithLogitsLoss를 손실 함수로 사용할 것이므로,
    모델 안에는 시그모이드를 넣지 않고 Linear 계층의 순수 출력(로짓, logit)을 그대로 반환한다.
    (시그모이드 계산은 손실 함수 내부에서 함께 처리되어 수치적으로 더 안정적)
    """
    def __init__(self, n_input, n_output):
        super().__init__()
        # 입력 2개(꽃받침 길이, 폭) → 출력 1개(로짓)로 매핑하는 선형 계층
        self.l1 = nn.Linear(n_input, n_output)

        # 초깃값을 모두 1.0으로 통일 (강의자료와 동일한 초기화 방식)
        # 실무에서는 보통 랜덤 초기화를 쓰지만, 학습 과정을 예측 가능하게 보기 위해 고정값을 사용
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        # 입력 x가 선형 계층(l1)을 통과한 결과(로짓)를 그대로 반환
        # 확률로 변환하는 시그모이드 계산은 아래 criterion(BCEWithLogitsLoss)이 대신 처리함
        x1 = self.l1(x)
        return x1


n_input = 2    # 입력 특성 개수: 꽃받침 길이, 꽃받침 폭
n_output = 1   # 출력 개수: 정답이 1일 확률에 대응하는 로짓 값
model = Net(n_input, n_output)


# ---------- 6. 손실 함수 및 옵티마이저 정의 ----------
# BCEWithLogitsLoss: 시그모이드 계산 + 교차 엔트로피 손실 계산을 하나로 합친 손실 함수
# 모델이 시그모이드 없이 로짓을 그대로 출력해도, 이 손실 함수가 내부에서 시그모이드까지 처리해줌
criterion = nn.BCEWithLogitsLoss()

# 확률적 경사 하강법(SGD) 옵티마이저
# lr(learning rate, 학습률)은 한 번의 업데이트에서 파라미터가 얼마나 이동할지를 결정하는 값
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)


# ---------- 7. 정확도 계산 함수 ----------
def calc_accuracy(model, X, y):
    """
    주어진 데이터(X, y)에 대한 모델의 정확도를 계산하는 함수

    BCEWithLogitsLoss를 사용하므로 예측 기준은 '로짓 값이 0보다 큰지'이며,
    이는 시그모이드 통과 후 확률이 0.5보다 큰 것과 수학적으로 동일한 의미이다.
    """
    with torch.no_grad():                       # 정확도 계산 시에는 경사(gradient)가 필요 없으므로 비활성화 → 연산 속도 향상
        logits = model(X)                        # 모델의 원본 출력(로짓)
        preds = (logits > 0.0).float()            # 로짓이 0보다 크면 1, 아니면 0으로 예측
        acc = (preds == y).float().mean().item()  # 예측이 정답과 일치하는 비율(정확도) 계산
    return acc


# ---------- 8. 학습 반복 (경사 하강법 5단계) ----------
num_epochs = 10000    # 전체 반복(epoch) 횟수

# 학습 곡선을 그리기 위해 기록해 둘 리스트들
recorded_epochs = []   # 기록이 이루어진 epoch 번호
train_loss_list = []   # 각 시점의 훈련 손실
val_loss_list = []     # 각 시점의 검증 손실
train_acc_list = []    # 각 시점의 훈련 정확도
val_acc_list = []      # 각 시점의 검증 정확도

for epoch in range(num_epochs):
    # [1단계] 경사값 초기화
    # 파이토치는 경사(gradient)를 기본적으로 계속 누적하므로,
    # 매 반복(epoch)마다 반드시 0으로 초기화해야 이전 반복의 경사가 섞이지 않는다
    optimizer.zero_grad()

    # [2단계] 순전파 (Forward)
    # 훈련 데이터를 모델에 통과시켜 예측값(로짓)을 계산
    outputs = model(X_train_t)

    # [3단계] 손실 계산
    # 예측값(로짓)과 실제 정답을 비교하여 교차 엔트로피 손실(BCEWithLogitsLoss)을 계산
    loss = criterion(outputs, y_train_t)

    # [4단계] 역전파 (Backward)
    # 손실을 기준으로 모델의 각 파라미터(가중치, 편향)에 대한 경사(기울기)를 계산
    loss.backward()

    # [5단계] 파라미터 업데이트
    # 계산된 경사를 바탕으로 옵티마이저가 가중치와 편향을 실제로 수정
    optimizer.step()

    # ---- 학습 곡선 기록 (100번마다 한 번씩만 기록하여 연산량 절약) ----
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        recorded_epochs.append(epoch)

        # 현재 시점의 훈련 손실 기록 (위에서 이미 계산된 loss 재사용)
        train_loss_list.append(loss.item())

        # 현재 시점의 검증 손실 계산 및 기록
        # 검증 데이터는 학습(파라미터 업데이트)에 전혀 사용하지 않고 평가에만 사용
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
        val_loss_list.append(val_loss.item())

        # 현재 시점의 훈련/검증 정확도 계산 및 기록
        train_acc_list.append(calc_accuracy(model, X_train_t, y_train_t))
        val_acc_list.append(calc_accuracy(model, X_val_t, y_val_t))


# ---------- 9. 학습 결과(최종 성능) 확인 ----------
final_train_acc = calc_accuracy(model, X_train_t, y_train_t)
final_val_acc = calc_accuracy(model, X_val_t, y_val_t)

# 학습이 끝난 후 최종적으로 학습된 파라미터(가중치, 편향) 확인
# 이 값들이 두 그룹을 나누는 결정 경계선의 위치와 기울기를 결정한다
print(f"학습된 파라미터 - WEIGHT: {model.l1.weight.data.numpy()}, BIAS: {model.l1.bias.data.numpy()}")
print(f"최종 훈련 정확도: {final_train_acc * 100:.1f}%")
print(f"최종 검증 정확도: {final_val_acc * 100:.1f}%")


# ---------- 10. 테스트 (검증 데이터에 대한 개별 예측 확인) ----------
with torch.no_grad():
    sample_logits = model(X_val_t[:5])            # 검증 데이터 중 앞 5개 샘플만 테스트
    sample_probs = torch.sigmoid(sample_logits)     # 로짓을 실제 확률(0~1)로 변환해 보기 쉽게 표시
    sample_preds = (sample_logits > 0.0).float()    # 최종 예측 결과(0 또는 1)

print("\n[검증 데이터 5개 샘플 테스트 결과]")
for i in range(5):
    print(f"실제 정답: {int(y_val_t[i].item())} | "
          f"예측 확률: {sample_probs[i].item():.3f} | "
          f"최종 예측: {int(sample_preds[i].item())}")


# ---------- 11. 반복 학습 횟수에 따른 손실 · 정확도 그래프 ----------
plt.figure(figsize=(12, 5))

# (왼쪽) 손실 곡선: 반복 횟수가 늘어날수록 손실이 줄어드는 모습을 확인
plt.subplot(1, 2, 1)
plt.plot(recorded_epochs, train_loss_list, label="훈련", color="blue")
plt.plot(recorded_epochs, val_loss_list, label="검증", color="black")
plt.xlabel("반복 횟수")
plt.ylabel("손실")
plt.title("학습 곡선 (손실)")
plt.legend()
plt.grid(True)

# (오른쪽) 정확도 곡선: 반복 횟수가 늘어날수록 정확도가 올라가는 모습을 확인
plt.subplot(1, 2, 2)
plt.plot(recorded_epochs, train_acc_list, label="훈련", color="blue")
plt.plot(recorded_epochs, val_acc_list, label="검증", color="black")
plt.xlabel("반복 횟수")
plt.ylabel("정확도")
plt.title("학습 곡선 (정확도)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("learning_curves.png")   # 그래프를 이미지 파일로 저장
plt.show()                            # 그래프를 화면에 출력
```

이 코드를 실행하면 초기 50% 수준이던 정확도가 학습을 거치며 **96.7%**까지 올라가고, 훈련·검증 곡선이 비슷하게 움직이는 것을 확인할 수 있다. 과적합 없이 잘 일반화되었다는 뜻이다.

---

## 💡 핵심 용어 4가지

* **이진 분류**: 두 그룹 중 하나로 나누는 작업
* **시그모이드 함수**: 출력을 0~1 확률로 변환
* **정확도**: 맞은 비율
* **교차 엔트로피**: 확신 있게 틀릴수록 큰 페널티를 주는 손실 함수
