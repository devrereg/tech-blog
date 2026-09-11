---
title: "신경망 스타일 전이(Neural Style Transfer) 완전 정복 — 원리부터 PyTorch 구현까지"
date: 2026-09-11 15:00:00 +0900
categories: [AI, Deep Learning]
tags: [neural-style-transfer, nst, vgg19, gram-matrix, lbfgs, pytorch, deep-learning, computer-vision, image-generation]
math: true
description: "사진 한 장과 그림 한 장을 합쳐 '고흐가 그린 우리 동네' 같은 이미지를 만드는 Neural Style Transfer의 원리를 파이프라인, 콘텐츠/스타일 손실과 α:β 가중치, Gram Matrix, PyTorch 구현, LBFGS 최적화, Fast Neural Style Transfer까지 정리했다."
---

> 사진 한 장과 그림 한 장을 합쳐서 "고흐가 그린 우리 동네" 같은 이미지를 만드는 기술 — Neural Style Transfer(NST)의 원리를 파이프라인부터 손실 함수, PyTorch 구현, 최적화 알고리즘까지 차근차근 정리합니다.
>
> [CNN 완전 정복]({% post_url 2026-08-28-cnn-convolution-complete-guide %})에서 합성곱 신경망이 이미지의 특징을 어떻게 추출하는지 다뤘고, [전이학습으로 VGG 활용하기]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})에서 사전학습된 VGG를 분류에 재사용하는 법을 살펴봤습니다. NST는 그 VGG를 분류가 아니라 **"이미지의 스타일과 내용을 뽑아내는 특징 추출기"**로 완전히 다른 방식으로 재활용하는 이야기입니다.

## 📌 목차

1. [스타일 전이란?](#what-is-nst)
2. [NST 전체 파이프라인](#pipeline)
3. [콘텐츠 손실과 스타일 손실, 뭐가 다를까?](#content-vs-style-loss)
4. [Gram Matrix: 화풍의 "지문"](#gram-matrix)
5. [PyTorch로 구현하기](#pytorch-impl)
   - [Gram Matrix 계산](#gram-matrix-calc)
   - [층별 가중치와 α:β 결합](#layer-weights)
   - [최적화 대상은 "이미지 자체"](#optimize-target)
   - [학습 루프](#training-loop)
   - [실전에서 신경 써야 할 것들](#practical-tips)
6. [왜 하필 LBFGS인가? (feat. 헤시안)](#lbfgs)
7. [마무리](#summary)

---

## 1. 스타일 전이란? {#what-is-nst}

스타일 전이(Style Transfer)는 한 문장으로 이렇게 정의할 수 있습니다.

> **이미지 C를 만들기 위해 이미지 B의 스타일을 이미지 A로 옮기는 것**

- **이미지 A (콘텐츠)**: 형태, 구도 같은 "무엇이 어디에 있는가" 정보를 담당
- **이미지 B (스타일)**: 붓터치, 색감, 질감 같은 "어떻게 그려졌는가" 정보를 담당
- **이미지 C (결과)**: 두 정보를 합쳐 새로 만들어진 이미지

일반적인 포토샵 합성과 다른 점은, 결과 이미지가 두 이미지를 오려 붙인 게 아니라 신경망이 "내용"과 "스타일"이라는 두 추상적 특징을 각각 뽑아내 **완전히 새로 재구성**한다는 점입니다.

이 아이디어는 2015년 Leon Gatys 등이 발표한 논문 [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576)에서 처음 제안됐습니다. "사전학습된 CNN의 중간 특징 맵만으로 콘텐츠와 스타일을 분리해낼 수 있다"는 이 논문의 통찰이 지금 다루는 NST 파이프라인 전체의 출발점입니다.

![콘텐츠 이미지와 스타일 이미지를 더하면 콘텐츠의 형태와 구도는 유지하면서 스타일의 색감과 질감이 입혀진 결과 이미지가 만들어진다는 개념도](/assets/img/posts/style_transfer/concept.svg)

*Neural Style Transfer = 콘텐츠의 형태와 스타일의 색감·질감을 하나로 합치는 작업*

---

## 2. NST 전체 파이프라인 {#pipeline}

NST가 특별한 이유는, 보통의 딥러닝과 반대로 동작하기 때문입니다.

| 일반적인 딥러닝 | NST |
|---|---|
| 데이터는 고정, **가중치**를 학습 | **가중치는 고정**(VGG19), 이미지 픽셀 자체를 학습 |

파이프라인은 다음과 같이 흘러갑니다.

![NST 파이프라인. 콘텐츠 이미지 p, 스타일 이미지 a, 생성 이미지 x가 각각 같은 VGG19를 통과해 특징 맵을 얻고, x-p로 콘텐츠 손실을, x-a로 스타일 손실을 계산해 합산한 총 손실 L_total을 경사하강으로 최소화하며 x를 반복 갱신하는 순환 구조](/assets/img/posts/style_transfer/pipeline.svg)

1. 스타일 이미지 `a`, 생성 이미지 `x`(처음엔 노이즈나 콘텐츠 이미지로 시작), 콘텐츠 이미지 `p` — 세 이미지를 준비
2. 셋 다 **같은 VGG19 신경망**(사전학습된 가중치를 고정한 채 특징 추출기로만 사용)을 통과
3. `x`와 `a`의 특징으로 **스타일 손실**, `x`와 `p`의 특징으로 **콘텐츠 손실** 계산
4. 두 손실을 가중합해 **총 손실**을 구함
5. **경사하강법**으로 `x`의 픽셀 값을 총 손실이 줄어드는 방향으로 수정
6. 2~5번을 수백 번 반복 → 콘텐츠는 유지하면서 스타일이 입혀진 이미지로 수렴

---

## 3. 콘텐츠 손실과 스타일 손실, 뭐가 다를까? {#content-vs-style-loss}

같은 VGG19를 통과하는데도 콘텐츠 손실과 스타일 손실이 서로 다른 정보를 잡아내는 이유는, **특징 맵을 비교하는 방식**이 다르기 때문입니다.

![콘텐츠 손실은 특징 맵의 같은 위치(i,j)끼리 직접 값을 빼서 비교하고, 스타일 손실은 특징 맵을 채널끼리 곱해 만든 Gram Matrix끼리 비교한다는 대조 다이어그램](/assets/img/posts/style_transfer/loss_compare.svg)

### 콘텐츠 손실

$$L_{content} = \frac{1}{2}\sum_{i,j}(F_{lij} - P_{lij})^2$$

생성 이미지의 특징 맵 `F`와 콘텐츠 이미지의 특징 맵 `P`를 **같은 위치(i, j)끼리 직접 비교**합니다. 위치 정보가 그대로 남기 때문에 "지붕이 어디 있고 강이 어디 있는지" 같은 **공간 배치(구도)**가 보존됩니다. 보통 conv4_2처럼 어느 정도 깊은 층 **1개**만 사용합니다 — 너무 얕은 층을 쓰면 색감·질감까지 원본에 강하게 묶여 스타일이 입혀질 여지가 없어지기 때문입니다.

### 스타일 손실

$$L_{style} = \sum_l w_l E_l, \quad E_l = \frac{1}{4N_l^2M_l^2}\sum(G^l_{ij} - A^l_{ij})^2$$

비교 대상이 특징 맵 자체가 아니라 **Gram Matrix**(`G`, `A`)라는 점이 다릅니다. Gram Matrix는 공간 위치를 다 더해서 없애버리고, 대신 "어떤 채널들이 서로 얼마나 자주 함께 활성화되는가"라는 통계만 남깁니다. 그래서 원본과 같은 위치일 필요 없이 질감/색감의 패턴만 비슷하면 되고, 질감은 미세한 붓터치부터 큰 색상 덩어리까지 여러 스케일에 걸쳐 있어서 보통 **여러 층**(conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 등)을 함께 사용합니다.

### 콘텐츠:스타일 비율(α:β)이 결과를 좌우한다

총 손실은 결국 두 손실의 가중합입니다.

$$L_{total} = \alpha L_{content} + \beta L_{style}$$

이 $\alpha$(콘텐츠 가중치)와 $\beta$(스타일 가중치)의 **비율**이 실제로 눈에 보이는 결과물 품질을 가장 크게 좌우하는 하이퍼파라미터입니다.

- $\alpha/\beta$가 크면(콘텐츠 비중↑): 원본 사진의 형태는 뚜렷이 남지만 스타일이 옅게 입혀짐
- $\alpha/\beta$가 작으면(스타일 비중↑): 화풍은 진하게 배지만 원본 구도가 무너지고 텍스처만 남을 수 있음

논문에서 제시한 대략적인 기준은 $\alpha/\beta = 10^{-3} \sim 10^{-4}$ 수준이며, 실제로는 이미지마다 원하는 스타일 강도가 다르므로 몇 가지 값을 직접 돌려보며 눈으로 확인하는 튜닝이 필요합니다.

---

## 4. Gram Matrix: 화풍의 "지문" {#gram-matrix}

Gram Matrix가 정확히 뭘 계산하는지 한 번 더 짚어보겠습니다.

$$G^l_{ij} = \sum_k F^l_{ik}F^l_{jk}$$

VGG19의 각 채널은 "노란색을 찾는 필터", "소용돌이 패턴을 찾는 필터", "직선 경계를 찾는 필터"처럼 저마다 특정 패턴에 반응하는 탐지기로 볼 수 있습니다. Gram Matrix는 이 탐지기들을 두 개씩 짝지어서, 같은 이미지 안에서 얼마나 자주 함께 반응하는지 궁합을 기록한 표입니다.

![붉은색 탐지기·초록색 탐지기·파란색 탐지기 세 필터가 서로 짝을 지어 값을 채우는 Gram Matrix 궁합표. 두 필터가 이미지 전체에서 함께 자주 활성화될수록 해당 칸의 값이 커진다는 개념도](/assets/img/posts/style_transfer/gram_matrix.svg)

두 탐지기가 이미지 전체에서 같이 자주 켜지면 그 칸의 값이 커지고, 따로 놀면 값이 작아집니다. 채널 수 × 채널 수 크기의 이 정사각 행렬 전체가 곧 그 이미지의 **화풍을 요약한 지문**이 되고, 스타일 손실은 결국 생성 이미지의 지문을 스타일 이미지의 지문에 맞춰가는 과정입니다.

---

## 5. PyTorch로 구현하기 {#pytorch-impl}

핵심 로직만 뽑아서 살펴보겠습니다.

### Gram Matrix 계산 {#gram-matrix-calc}

```python
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)          # 공간(h,w)을 하나의 축으로 평탄화
        G = torch.bmm(F, F.transpose(1, 2))   # 채널끼리 내적 → (b, c, c)
        G.div_(h * w)                          # 이미지 크기로 정규화
        return G
```

`torch.bmm(F, F.transpose(1, 2))` 이 한 줄이 앞서 본 수식 $G^l_{ij} = \sum_k F^l_{ik}F^l_{jk}$ 그 자체입니다.

스타일 손실은 특징 맵이 아니라 Gram Matrix끼리 비교하므로, MSELoss를 적용하기 전에 Gram Matrix로 변환하는 래퍼가 하나 필요합니다.

```python
class GramMSELoss(nn.Module):
    def forward(self, input, target):
        G = GramMatrix()(input)          # 생성 이미지 특징 맵 → Gram Matrix
        return nn.MSELoss()(G, target)   # 스타일 이미지의 Gram Matrix(target)와 비교
```

### 층별 가중치와 α:β 결합 {#layer-weights}

```python
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']  # 여러 층
content_layers = ['r42']                              # 1개 층
loss_layers = style_layers + content_layers

loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

# w_l: 층별 가중치. 채널 수가 많은 깊은 층일수록 가중치를 작게 → 층별 손실 크기를 정규화
style_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
content_weight = 1e0  # α: 콘텐츠 손실 전역 가중치

# 최종 weights = [스타일 층별 가중치들, 콘텐츠 가중치] — 3장의 α:β는 이 리스트 전체의 균형에 해당
weights = style_weights + [content_weight]

# 콘텐츠/스타일 이미지를 한 번만 VGG에 통과시켜 "목표(target)" 특징 맵을 미리 뽑아둔다
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets
```

여기서 `style_weights`(층별 가중치 $w_l$)와 3장에서 본 전역 가중치 α, β는 서로 다른 층위입니다. $w_l$은 "스타일 손실 안에서 어느 층을 더 믿을지"를 정하고, α·β는 "콘텐츠 손실과 스타일 손실 두 그룹 사이의 균형"을 정합니다. 위 코드에서는 `content_weight`가 α 역할을, `style_weights`의 스케일(`1e3`)이 사실상 β 역할을 겸하고 있습니다.

`targets`를 반복문 밖에서 미리 한 번만 계산해두는 이유는, 콘텐츠 이미지와 스타일 이미지는 최적화 대상이 아니라 **고정된 목표값**이기 때문입니다 — 매 반복마다 다시 계산할 필요가 없습니다.

### 최적화 대상은 "이미지 자체" {#optimize-target}

```python
input_image = content_image.clone().requires_grad_(True)
optimizer = torch.optim.LBFGS([input_image], max_iter=1)
```

`requires_grad_(True)`가 붙은 대상이 **모델 가중치가 아니라 이미지 텐서**라는 점이 NST를 이해하는 가장 중요한 포인트입니다.

### 학습 루프 {#training-loop}

```python
def closure():
    optimizer.zero_grad()
    out = vgg(input_image, loss_layers)

    total_loss = 0
    for i, weight in enumerate(weights):
        loss = weight * loss_fns[i](out[i], targets[i])
        total_loss += loss

    total_loss.backward()
    return total_loss

for i in range(num_iterations):
    optimizer.step(closure)
```

매 반복마다 생성 이미지 `input_image`만 VGG(`vgg(input_image, loss_layers)`)에 새로 통과시켜 `out`을 얻고, 미리 뽑아둔 `targets`와 비교해 손실을 계산합니다. 역전파로 얻은 기울기만큼 이미지 픽셀을 조금씩 수정하는 과정이 `num_iterations`번 반복됩니다.

### 실전에서 신경 써야 할 것들 {#practical-tips}

- **입력 정규화**: VGG19는 ImageNet의 채널별 평균/표준편차로 정규화된 입력을 학습했으므로, 콘텐츠·스타일·생성 이미지 모두 같은 mean/std로 정규화해서 VGG에 넣어야 특징 맵이 의미 있게 나옵니다.
- **초기화 전략**: `input_image`를 무작위 노이즈로 시작할 수도, 콘텐츠 이미지 복사본으로 시작할 수도 있습니다. 노이즈 초기화는 더 자유롭게 스타일이 입혀지지만 수렴이 느리고 불안정하며, 콘텐츠 이미지로 초기화하면 훨씬 빠르고 안정적으로 수렴합니다 — 그래서 실무에서는 후자를 더 많이 씁니다.
- **이미지 크기**: 특징 맵 크기가 곧 계산량이므로, 처음엔 작은 해상도(256~512px)로 빠르게 결과를 확인한 뒤 고해상도로 올리는 순서가 효율적입니다.

---

## 6. 왜 하필 LBFGS인가? (feat. 헤시안) {#lbfgs}

일반적인 경사하강법은 **1차 미분(기울기)**만 사용합니다.

$$x := x - \lambda\frac{\partial L_{total}}{\partial x}$$

문제는 손실 함수의 표면이 길쭉하고 울퉁불퉁할 때, 1차 미분만으로는 **지그재그로 비효율적으로** 내려간다는 점입니다. 반면 **헤시안(Hessian) 행렬**, 즉 2차 미분들을 모아놓은 행렬을 활용하면 "이 방향은 완만하고 저 방향은 가파르다"는 **곡률** 정보까지 알 수 있어 훨씬 적은 단계로 최적점에 도달할 수 있습니다.

![경사하강법은 타원형 등고선을 따라 지그재그로 꺾이며 내려가는 반면, 헤시안 기반 최적화(LBFGS)는 곡률 정보를 활용해 더 적은 단계로 직선에 가깝게 최적점에 도달한다는 경로 비교도](/assets/img/posts/style_transfer/optimization.svg)

`torch.optim.LBFGS`는 진짜 헤시안 전체를 계산하진 않지만(변수가 많으면 계산량이 너무 커지므로), **최근 몇 번의 기울기 변화 이력으로 곡률을 근사**합니다. 그래서:

- 일반 딥러닝(가중치 수백만~수십억 개, 배치마다 데이터가 바뀌는 확률적 환경)에는 잘 안 맞지만
- NST처럼 **이미지 1장**만 최적화하고 데이터가 고정된 **결정론적** 문제에는 아주 잘 맞습니다 — 같은 반복 횟수로도 더 선명하고 안정적인 결과를 얻을 수 있습니다.

### 한계: 그래도 느리다

LBFGS를 쓰더라도 이미지 한 장을 만들기 위해 VGG 순전파·역전파를 수백 번 반복해야 하므로, 지금까지 본 방식(원 논문 기준 "Gatys 방식" 또는 "최적화 기반 NST")은 이미지 한 장에 수십 초~수 분이 걸립니다. 실시간 서비스에는 부적합하다는 뜻입니다.

이 한계를 해결한 것이 Johnson et al.(2016)의 **Fast Neural Style Transfer**입니다. 스타일마다 이미지 변환 신경망을 미리 학습시켜두면, 이후에는 이미지를 그 네트워크에 **한 번만 순전파**시켜 즉시 결과를 얻을 수 있습니다 — "이미지를 최적화"하던 문제를 "변환 함수를 학습"하는 문제로 바꾼 것입니다. 지금까지 다룬 Gatys 방식의 손실 함수(콘텐츠 손실 + 스타일 손실)는 그대로 재사용하되, 학습 대상만 이미지 픽셀에서 신경망 가중치로 옮겨간 셈입니다.

---

## 7. 마무리 {#summary}

- NST는 콘텐츠 이미지와 스타일 이미지를 합쳐 새 이미지를 만드는 기술
- 콘텐츠는 VGG19의 **1개 층**에서 특징 맵을 직접 비교 → 위치/구도 보존
- 스타일은 VGG19의 **여러 층**에서 **Gram Matrix**(채널 간 상관관계)를 비교 → 질감/색감 통계 보존
- 총 손실 = 콘텐츠 손실 + 스타일 손실의 가중합
- 학습 대상은 모델 가중치가 아니라 **이미지 픽셀 자체**
- 콘텐츠:스타일 가중치 비율(α:β)이 결과물의 스타일 강도를 좌우
- 최적화에는 곡률(헤시안) 정보를 활용하는 **LBFGS**가 적합하지만, 이미지 1장에 수십 초~수 분이 걸릴 만큼 느리다

**핵심 체크리스트**

- [ ] NST는 가중치가 아니라 이미지 픽셀을 학습 대상으로 삼는다
- [ ] 콘텐츠 손실은 특징 맵을 같은 위치끼리 직접 비교한다 (1개 층)
- [ ] 스타일 손실은 Gram Matrix(채널 간 상관관계)를 비교한다 (여러 층)
- [ ] Gram Matrix는 공간 위치를 없애고 "어떤 채널이 함께 활성화되는가"만 남긴다
- [ ] α(콘텐츠 가중치)와 β(스타일 가중치)의 비율로 스타일 강도를 조절한다
- [ ] `input_image.requires_grad_(True)`가 NST의 핵심 — 모델이 아니라 이미지가 학습된다
- [ ] LBFGS는 곡률(2차 미분) 정보를 근사해 결정론적 단일 이미지 최적화에 잘 맞는다
- [ ] Gatys 방식은 느리다는 한계가 있고, Fast Neural Style Transfer는 이를 "이미지 최적화"에서 "변환 신경망 학습"으로 바꿔 해결한다

---

*이 글은 Gatys et al.(2015)의 Neural Style Transfer 논문과 PyTorch 실습 코드를 바탕으로 원리를 정리한 학습 노트입니다.*
