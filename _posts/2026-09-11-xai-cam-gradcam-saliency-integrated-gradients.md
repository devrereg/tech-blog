---
title: "설명 가능한 AI(XAI) 핵심 4대 기법 완전 정리: CAM부터 Integrated Gradients까지"
date: 2026-09-11 03:00:00 +0900
categories: [AI, Deep Learning]
tags: [xai, explainable-ai, cam, grad-cam, saliency-map, integrated-gradients, cnn, interpretability, deep-learning, computer-vision]
math: true
description: "딥러닝 모델이 '왜' 그렇게 판단했는지 시각적으로 설명하는 4가지 기법 — CAM, Grad-CAM, Saliency Map, Integrated Gradients — 을 원리부터 계산 과정, 실전 선택 기준까지 정리했다."
---

> 딥러닝 모델은 흔히 **블랙박스**라고 불립니다. 입력과 출력은 확인할 수 있지만, 내부에서 어떤 근거로 그 결론에 도달했는지는 알기 어렵습니다. 이 글에서는 모델의 판단 근거를 히트맵으로 시각화하는 4가지 기법 — **CAM, Grad-CAM, Saliency Map, Integrated Gradients** — 을 원리부터 계산 과정, 선택 기준까지 하나씩 정리합니다.
>
> [CNN 완전 정복]({% post_url 2026-08-28-cnn-convolution-complete-guide %})에서 다룬 특징맵(Feature Map)과 역전파(Backpropagation) 개념이 이 네 기법 모두의 계산 근간이 됩니다.

## 📌 목차

1. [왜 XAI가 필요한가](#why-xai)
2. [CAM (Class Activation Mapping)](#cam)
3. [Grad-CAM](#gradcam)
4. [Saliency Map](#saliency-map)
5. [Integrated Gradients](#integrated-gradients)
6. [네 기법 종합 비교](#comparison)
7. [마무리 — 흔한 오해 하나](#summary)

---

## 1. 왜 XAI가 필요한가 {#why-xai}

- **의료 진단**: "왜 암이라고 판단했나요?" — 의사가 병변 위치를 확인해야 진단을 신뢰할 수 있습니다.
- **대출 심사**: "왜 거절됐나요?" — 투명한 사유가 있어야 고객이 이의를 제기할 수 있습니다.
- **자율주행**: "왜 급정거했나요?" — 사고 발생 시 원인 추적과 책임 소재 규명에 필요합니다.

이 세 사례의 공통점은, **결과(What)만으로는 부족하고 과정(Why)이 있어야 신뢰(Trust)가 생긴다**는 것입니다.

```
결과(What) + 이유(Why) = 신뢰(Trust)
```

**XAI(eXplainable AI)** 는 이 "이유"를 시각적으로 보여주는 기술입니다. 이미지 분류 모델을 예로 들면:

```
입력: [이미지] → [모델] → 결과: 강아지 20%, 고양이 80%
```

이 결과가 왜 나왔는지, 이미지의 어느 부분을 근거로 판단했는지 **히트맵**으로 시각화하는 것이 이 글에서 다루는 4가지 기법의 공통 목적입니다. 계산 방식만 서로 다를 뿐입니다.

---

## 2. CAM (Class Activation Mapping) {#cam}

### 핵심 아이디어

CAM은 CNN 마지막 층의 **특징맵(Feature Map)** 에, 모델이 학습 과정에서 얻은 **가중치**를 곱해 위치 정보를 복원하는 기법입니다.

CNN이 이미지를 처리하면 마지막 합성곱층에서 여러 장(예: 512장)의 특징맵이 나옵니다. 각 특징맵은 7×7 크기의 격자로, "어떤 패턴이 이미지의 어느 위치에 있는지"를 나타냅니다.

일반적인 분류 모델은 이 특징맵들을 **Global Average Pooling(GAP)** 으로 평균 내어 숫자 하나씩으로 압축한 뒤, 완전연결층(FC layer)의 가중치를 곱해 클래스별 점수를 계산합니다. CAM은 이 가중치를 "빌려와" 원래의 7×7 특징맵에 다시 곱하는 방식으로 위치 정보를 복원합니다.

### 계산 과정

1. 특징맵 512장(각 7×7)에 대해, 학습된 가중치 $W_k$를 각각 곱함
2. 512개의 결과를 위치(칸)별로 모두 더함 → 7×7 결과 지도 1장
3. 이 거친 7×7 지도를 **양선형 보간(Bilinear Interpolation)** 으로 원본 이미지 크기로 확대 → 히트맵 완성

![CNN 마지막 합성곱층에서 나온 512장의 7×7 특징맵 각각에 학습된 가중치 W_k를 곱하고, 그 결과들을 위치별로 모두 더해 7×7 결과 지도를 만든 뒤 양선형 보간으로 원본 이미지 크기로 확대해 히트맵을 얻는 CAM 파이프라인 흐름도](/assets/img/posts/xai/cam-pipeline.svg)

*512장의 7×7 특징맵에 GAP 이전에 학습된 가중치를 곱해 더하면, 모델이 어디를 보고 판단했는지 나타내는 거친 히트맵이 나온다*

### 코드로 보면

```python
import torch.nn.functional as F

feature_map = features(image)                    # (512, 7, 7) — 마지막 합성곱층 출력
weights = fc_layer.weight[class_idx]              # (512,) — GAP 뒤 FC layer의 학습된 가중치

cam = torch.einsum("c,chw->hw", weights, feature_map)          # (7, 7)
cam = F.interpolate(cam[None, None], size=image.shape[-2:],
                     mode="bilinear")[0, 0]                    # 원본 크기로 확대
```

### 한계

CAM은 **GAP 레이어가 반드시 있어야만** 작동합니다. 모델이 GAP 대신 다른 구조(예: 여러 개의 FC layer)를 쓰고 있다면 적용이 불가능하고, 구조를 바꿔 재학습해야 합니다. 이 한계를 극복한 것이 Grad-CAM입니다.

---

## 3. Grad-CAM {#gradcam}

### 핵심 아이디어

Grad-CAM은 CAM의 "GAP 가중치" 대신 **기울기(Gradient)** 를 사용합니다. 기울기는 "이 특징맵이 조금 변하면 최종 점수가 얼마나 민감하게 변하는가"를 나타내는 값으로, GAP 구조 없이도 **역전파(Backpropagation)** 만 가능하면 어떤 CNN에서든 계산할 수 있습니다.

> **가중치(Weight)** 는 학습이 끝난 뒤 고정된 값이고, **기울기(Gradient)** 는 입력 이미지마다 매번 새로 계산되는 민감도 값입니다. 학습 시의 역전파는 오차(Loss)를 기준으로 가중치를 실제로 수정하지만, Grad-CAM의 역전파는 모델 출력 점수($y^c$)를 기준으로 하며 가중치를 전혀 수정하지 않고 값만 읽어서 사용합니다.

### 계산 과정

1. 관심 클래스의 출력 점수 $y^c$(softmax 직전 값)를 기준으로, 마지막 합성곱층의 각 특징맵 $A^k$에 대해 역전파로 기울기 계산 (7×7 형태로 나옴)
2. 이 7×7 기울기를 공간적으로 평균 내어 숫자 하나(**$\alpha^k$**, importance weight)로 압축 — 이 과정을 512개 특징맵 전부에 대해 반복
3. $\alpha^k \times A^k$(원본 7×7 특징맵 전체)를 모든 $k$에 대해 계산 후 위치별로 가중합 → 7×7 결과
4. **ReLU**를 적용해 음수(판단을 방해한 요소)는 제거하고 양수(긍정적 근거)만 남김
5. 보간법으로 원본 이미지 크기로 확대 → 히트맵 완성

![관심 클래스 출력 점수로부터 역전파해 마지막 합성곱층의 각 7×7 특징맵에 대한 기울기를 구하고, 이를 공간 평균 내어 얻은 중요도 가중치 α^k를 원본 특징맵에 곱해 더한 뒤 ReLU로 음수를 제거하고 보간으로 확대하는 Grad-CAM 파이프라인 흐름도](/assets/img/posts/xai/gradcam-pipeline.svg)

*GAP 가중치 대신 기울기의 공간 평균(α^k)을 중요도로 쓰고, ReLU로 긍정적 근거만 남기는 것이 Grad-CAM의 핵심이다*

### 코드로 보면

forward/backward hook으로 마지막 합성곱층의 특징맵과 기울기를 붙잡아 두면 됩니다.

```python
activations, gradients = {}, {}

target_layer.register_forward_hook(
    lambda m, i, o: activations.setdefault("A", o))
target_layer.register_full_backward_hook(
    lambda m, gi, go: gradients.setdefault("dA", go[0]))

score = model(image)[0, class_idx]
model.zero_grad()
score.backward()

A = activations["A"][0]                  # (512, 7, 7)
dA = gradients["dA"][0]                  # (512, 7, 7)
alpha = dA.mean(dim=(1, 2))              # (512,) — 공간 평균 = 중요도

cam = F.relu(torch.einsum("c,chw->hw", alpha, A))               # ReLU로 양수만
cam = F.interpolate(cam[None, None], size=image.shape[-2:],
                     mode="bilinear")[0, 0]
```

### CAM vs Grad-CAM

| 항목 | CAM | Grad-CAM |
|---|---|---|
| 구조 조건 | GAP 레이어 필수 | 구조 제약 없음 |
| 핵심 값 | 학습된 가중치 | 기울기의 공간 평균($\alpha^k$) |
| 값이 정해지는 시점 | 학습 시 고정 | 이미지 입력마다 새로 계산 |

> 두 기법 모두 히트맵의 해상도(선명도)는 마지막 합성곱층 특징맵의 크기(예: 7×7)에 좌우된다는 공통 한계를 가진다. 원본 해상도만큼 세밀하게 보고 싶다면 뒤에서 다룰 Saliency Map이나 Integrated Gradients가 필요하다.

---

## 4. Saliency Map {#saliency-map}

### 핵심 아이디어

Saliency Map은 "특징맵 단위"가 아니라 **입력 이미지의 픽셀 하나하나**에 대해 직접 민감도를 측정합니다. 핵심 질문은 다음과 같습니다.

> "입력 이미지의 픽셀을 아주 조금 바꿨을 때, AI의 예측 결과가 얼마나 크게 변하는가?"

Grad-CAM이 역전파를 "특징맵 지점까지만" 읽어냈다면, Saliency Map은 역전파를 **입력 픽셀까지 끝까지** 진행해서 그 지점의 기울기를 읽어냅니다. 계산 원리(연쇄법칙 기반 역전파) 자체는 동일하고, "어디의 기울기를 꺼내 쓰느냐"만 다릅니다.

### 계산 과정

1. 원본 이미지 1장을 모델에 입력
2. 순전파로 관심 클래스 점수 $y^c$를 얻음
3. $y^c$를 기준으로 역전파 1회 — 입력층(픽셀)까지 도달하여, 픽셀 개수만큼(예: 224×224개)의 기울기 값을 얻음
4. 이 기울기 값 자체(절댓값)를 이미지 형태로 배치 → 히트맵 완성 (별도의 곱셈이나 Baseline 없음)

![원본 이미지를 모델에 입력해 순전파로 클래스 점수를 얻고, 그 점수를 기준으로 역전파를 입력 픽셀까지 끝까지 진행해 픽셀 개수만큼의 기울기 값을 얻은 뒤 절댓값을 이미지 형태로 배치해 히트맵을 만드는 Saliency Map 파이프라인 흐름도](/assets/img/posts/xai/saliency-pipeline.svg)

*역전파를 특징맵이 아니라 입력 픽셀까지 끝까지 진행하면, 픽셀 단위 민감도 지도를 별도 연산 없이 바로 얻을 수 있다*

### 코드로 보면

```python
image.requires_grad_(True)

score = model(image)[0, class_idx]
model.zero_grad()
score.backward()

saliency = image.grad.abs().amax(dim=1)[0]   # 채널(RGB) 중 최댓값 → (H, W)
```

### 장점과 단점

| 장점 | 단점 |
|---|---|
| 역전파 1회 계산만으로 즉시 결과를 얻음 (빠름) | 결과물이 노이즈처럼 산만하게 보일 수 있음 |
| 픽셀 단위로 세밀하게 확인 가능 | **기울기 소실/포화**: 모델이 이미 확신에 찬 구간에서는 중요한 픽셀도 기울기가 0으로 표시될 수 있음 |

> ⚠️ **흔한 오해**: "Saliency Map이 선명할수록 모델 성능이 좋다"는 것은 사실이 아닙니다. 오히려 극히 일부 특징에만 뾰족하게 반응하는 것은 모델의 과적합(예: 배경 워터마크에만 의존) 신호일 수 있습니다.

Saliency Map의 "포화 문제"를 해결하기 위해 등장한 것이 Integrated Gradients입니다.

---

## 5. Integrated Gradients {#integrated-gradients}

### 핵심 아이디어

> 한 지점의 기울기만 보는 대신, **출발점(Baseline)부터 도착점(원본 이미지)까지의 모든 기울기를 누적한다.**

Saliency Map은 원본 이미지 딱 한 곳에서만 기울기를 측정했기 때문에, 그 지점이 "이미 판단이 끝나 평평해진 구간(포화 구간)"이면 중요한 픽셀도 기울기 0으로 잘못 표시될 수 있었습니다. Integrated Gradients는 **여러 지점에서 여러 번** 측정해서 이 문제를 완화합니다.

### Baseline이란?

Baseline은 "정보가 전혀 없는 상태"를 나타내는 인위적인 기준 이미지로, 보통 완전히 검은 이미지(모든 픽셀값 = 0)를 사용합니다. 텍스트 모델이라면 의미 없는 빈 토큰(패딩)을 Baseline으로 씁니다. Baseline은 반드시 "정보가 없는" 상태를 대표해야 하며, 잘못 설정하면 기여도 해석이 왜곡될 수 있습니다.

### 계산 과정

1. **경로 이미지 준비**: Baseline($\alpha=0$, 완전 검정)과 원본($\alpha=1$) 사이를 선형 보간으로 여러 단계로 나눕니다.

   $$\text{중간 이미지} = \text{Baseline} + \alpha \times (\text{원본 이미지} - \text{Baseline})$$

   Baseline이 완전 검정(0)이라면 `중간 이미지 = α × 원본 픽셀값`이 되어, 모든 픽셀이 검정에서 원본 색으로 동시에, 같은 비율로 "페이드인"됩니다.
2. **각 이미지마다 Saliency Map 계산**: 준비된 이미지(보통 수십~수백 장) 각각에 대해 순전파+역전파를 반복하여 픽셀별 기울기 지도를 구합니다.
3. **평균**: 여러 장의 기울기 지도를 위치별로 평균 냅니다.
4. **곱하기**: 평균 기울기에 `(원본 이미지 − Baseline)`을 곱해야 비로소 최종 히트맵이 완성됩니다. 이 곱셈은 "그 픽셀이 실제로 얼마나 값이 변했는지"를 반영하기 위해 반드시 필요합니다.

![Baseline(완전 검정 이미지)과 원본 이미지 사이를 선형 보간으로 여러 단계의 경로 이미지로 나누고, 각 이미지마다 Saliency Map을 계산한 뒤 위치별로 평균 내고, 마지막으로 (원본 − Baseline)을 곱해 최종 히트맵을 완성하는 Integrated Gradients 파이프라인 흐름도](/assets/img/posts/xai/ig-pipeline.svg)

*Baseline에서 원본까지 여러 지점의 기울기를 누적 평균하면, 한 지점만 볼 때 생기는 포화 문제를 완화할 수 있다*

### 코드로 보면

직접 구현할 수도 있지만, 실무에서는 보통 `captum` 라이브러리를 씁니다.

```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
baseline = torch.zeros_like(image)                 # 완전 검정 이미지

attributions = ig.attribute(image, baseline,
                             target=class_idx, n_steps=50)   # 경로를 50단계로 분할
heatmap = attributions[0].abs().sum(dim=0)                   # (H, W)
```

`n_steps`가 곧 "Baseline과 원본 사이를 몇 단계로 나누는가"이며, 값이 클수록 근사가 정교해지는 대신 순전파+역전파 횟수도 그만큼 늘어납니다.

### 완전성(Completeness) — 케이크 비유

Integrated Gradients는 다른 기법에 없는 수학적 보장을 갖습니다.

> 재료(각 픽셀)의 기여도를 모두 더하면, 결국 케이크 전체(모델 출력)의 맛 점수가 되어야 한다.

이를 수식으로 쓰면 다음과 같습니다.

$$\sum (\text{모든 픽셀의 기여도}) = f(\text{원본 이미지}) - f(\text{Baseline 이미지})$$

즉, 모든 픽셀의 히트맵 값을 다 더하면 정확히 "원본 점수 − Baseline 점수"와 일치해야 합니다. 이는 Saliency Map이나 CAM/Grad-CAM에는 없는, 수학적으로 검증 가능한 성질입니다.

### 장점과 단점

| 장점 | 단점 |
|---|---|
| 기울기 포화 문제 해결 | 계산 비용이 높음 (이미지 수십~수백 장 반복 계산) |
| 완전성(Completeness) 이론적 보장 | 적절한 Baseline 설정이 필요 |
| Saliency Map보다 노이즈 적음 | |

---

## 6. 네 기법 종합 비교 {#comparison}

| 기법 | 계산 단위 | 필요 조건 | 계산 속도 | 설명 신뢰도 | 적합한 상황 |
|---|---|---|---|---|---|
| **CAM** | 특징맵(7×7) | GAP 구조 필수 | 보통 | 보통 | 간단한 CNN 구조, 대략적 위치 파악 |
| **Grad-CAM** | 특징맵(7×7) | 구조 제약 없음 | 보통 | 보통 | 이미 학습된 다양한 모델, 일반적 시각화 |
| **Saliency Map** | 픽셀(원본 해상도) | 역전파만 가능하면 됨 | 빠름 | 낮음 | 빠른 모델 디버깅, 실시간 처리 |
| **Integrated Gradients** | 픽셀(원본 해상도) | Baseline 설정 필요 | 느림 | 높음 | 금융/의료 등 고신뢰성, 규제 대응 |

### 기법 선택 흐름

```
Q1. 이미지(CNN) 데이터인가?
 ├─ No (정형/텍스트) → 실시간 속도가 중요한가?
 │                       ├─ Yes → 규제/신뢰 요구가 높은가?
 │                       │          ├─ Yes → Integrated Gradients
 │                       │          └─ No  → Saliency Map
 │                       └─ No  → Integrated Gradients
 └─ Yes (이미지) → 위치(어디를 봤는지) 확인이 중요한가?
                     ├─ Yes → GAP 구조가 있거나 수정 가능한가?
                     │          ├─ Yes → CAM
                     │          └─ No  → Grad-CAM
                     └─ No (정밀 분석 필요) → 규제/신뢰 요구가 높은가?
                                              ├─ Yes → Integrated Gradients
                                              └─ No  → Saliency Map
```

핵심은 "어떤 기법이 가장 좋은가"가 아니라, **"지금 내 상황(데이터 종류, 목적, 속도·신뢰 제약)에 뭐가 필요한가"** 를 먼저 묻는 것입니다.

---

## 7. 마무리 — 흔한 오해 하나 {#summary}

> **오해**: "설명이 그럴싸하면 정답이다."
> **사실**: 아닙니다. XAI는 모델이 **틀린 이유도** 설명할 수 있습니다. 히트맵은 모델의 "생각"을 보여줄 뿐, 그 생각이 옳다는 것을 보장하지 않습니다.

예를 들어 배경의 워터마크만 보고 항상 "고양이"라고 판단하도록 잘못 학습된 모델이 있다면, 히트맵은 그 워터마크 부분을 아주 선명하게 강조할 것입니다. 이는 모델이 올바르게 판단했다는 뜻이 아니라, 오히려 **잘못된 편향(bias)을 발견**한 것에 가깝습니다.

XAI는 문제를 "해결"해주는 도구가 아니라 "발견"하게 해주는 도구입니다. CAM, Grad-CAM, Saliency Map, Integrated Gradients는 모두 이 발견의 과정을, 각자 다른 계산 방식으로 돕는 도구들입니다.

**핵심 체크리스트**

- [ ] CAM은 GAP 레이어의 가중치를 특징맵에 다시 곱해 위치를 복원한다
- [ ] Grad-CAM은 가중치 대신 기울기의 공간 평균(α^k)을 써서 구조 제약 없이 동작한다
- [ ] Saliency Map은 역전파를 입력 픽셀까지 끝까지 진행해 픽셀 단위 민감도를 얻는다
- [ ] Saliency Map은 빠르지만 기울기 포화 구간에서 중요한 픽셀도 0으로 표시될 수 있다
- [ ] Integrated Gradients는 Baseline부터 원본까지 여러 지점의 기울기를 누적 평균한다
- [ ] Integrated Gradients만 완전성(Completeness)이 수학적으로 보장된다
- [ ] 히트맵이 선명하다고 모델이 옳게 판단했다는 뜻은 아니다 — 편향을 드러내는 신호일 수도 있다

**참고: 4대 기법 핵심 키워드 한 줄 요약**

- **CAM / Grad-CAM** — "이미지의 **어디**를 봤나?" (위치 시각화)
- **Saliency Map** — "얼마나 **민감**하게 반응하나?" (픽셀 단위 민감도)
- **Integrated Gradients** — "**과정 전체**를 누적했나?" (경로 기반 누적 기여도)

---

*이 글은 CAM, Grad-CAM, Saliency Map, Integrated Gradients의 원리를 개념부터 계산 과정까지 정리한 학습 노트입니다.*
