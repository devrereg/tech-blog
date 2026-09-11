---
title: "AI는 새로운 이미지를 어떻게 만들까: 생성형 이미지 AI(GAN, VAE) 원리 총정리"
date: 2026-09-09 02:30:00 +0900
categories: [AI, Deep Learning]
tags: [generative-model, gan, dcgan, pix2pix, vae, autoencoder, deep-learning, pytorch, image-generation, machine-learning]
math: true
description: "판별하는 AI가 아니라 창조하는 AI, 즉 생성형 모델의 원리를 정리했다. '데이터의 분포를 학습해 새로운 샘플을 만든다'는 하나의 목표를, 경쟁으로 학습하는 GAN·DCGAN·Pix2Pix 계열과 확률 분포로 직접 모델링하는 VAE라는 두 갈래로 나눠 개념부터 구조, 수식, 코드까지 짚는다."
---

> 우리가 흔히 접하는 AI는 대부분 "판별하는" AI입니다. 사진을 보고 고양이인지 아닌지 분류하고, 사진 속 물체의 위치를 찾아내죠. 그런데 최근 몇 년 사이 "판단"이 아니라 "창조"를 하는 AI가 등장했습니다. 이 글 하나로 생성형 이미지 AI를 관통하는 두 축 **GAN**과 **VAE**의 핵심 원리를, 개념부터 구조와 수식, 코드까지 정리합니다.
>
> [CNN 완전 정복]({% post_url 2026-08-28-cnn-convolution-complete-guide %})에서 합성곱이 이미지의 공간 구조를 어떻게 살리는지 다뤘습니다. 이번 글의 DCGAN·Pix2Pix는 그 합성곱 블록을 "이미지를 분류하는" 방향이 아니라 "이미지를 만들어내는" 방향으로 뒤집어 쓰는 이야기입니다.

## 📌 목차

1. [판별형 모델 vs 생성형 모델](#discriminative-vs-generative)
2. [생성형 모델이란: "분포"를 학습한다](#what-is-generative)
3. [GAN — 경쟁을 통한 학습](#gan)
4. [DCGAN — 구조로 안정성을 잡다](#dcgan)
5. [GAN의 확장: SRGAN, CycleGAN, Pix2Pix](#gan-variants)
6. [VAE — 확률로 접근하는 생성](#vae)
7. [GAN vs VAE 최종 비교](#gan-vs-vae)
8. [마무리](#summary)

---

## 1. 판별형 모델 vs 생성형 모델 {#discriminative-vs-generative}

머신러닝 모델은 크게 두 부류로 나눌 수 있습니다.

- **판별형 모델(Discriminative Model)**: 입력 $x$가 주어졌을 때 정답 $y$를 맞히는 모델. 조건부 확률 $p(y \mid x)$를 학습합니다. 분류(Classification), 검출(Detection), 회귀가 모두 여기에 속합니다.
- **생성형 모델(Generative Model)**: 데이터 $x$ 자체가 어떻게 생겼는지, 즉 $p(x)$(또는 $p(x \mid y)$)를 학습하는 모델. 학습한 분포에서 표본을 뽑으면 그것이 **존재하지 않던 새 데이터**가 됩니다.

![상단에 'AI는 새로운 이미지를 만들어낼 수 있을까?'라는 Key Question이 있고, 그 아래 기존 AI(Discriminative, 분류·검출로 판단에 집중)와 생성형 AI(Generative, 생성·합성으로 창조에 집중) 두 카드를 나란히 비교하며, 하단에는 "판단이 아니라 창조"라는 방향의 차이를 요약한 어두운 색 박스가 있는 슬라이드](/assets/img/posts/generative-ai/01_generative_intro_question.svg)

*판별형은 "이 이미지가 무엇인가"를 답하고, 생성형은 "이런 이미지는 대략 어떻게 생겼는가"를 배운 뒤 새로 그려낸다*

---

## 2. 생성형 모델이란: "분포"를 학습한다 {#what-is-generative}

생성형 모델을 한 문장으로 정의하면 다음과 같습니다.

> **생성형 모델은 데이터의 분포를 학습하여, 존재하지 않는 새로운 데이터를 만들어내는 모델이다.**

여기서 핵심 키워드는 **분포(Distribution)** 입니다. AI가 이미지 하나하나를 통째로 암기하는 게 아니라, "이런 이미지들이 대략 어떤 패턴과 범위 안에 존재하는가"라는 **확률적 패턴 자체**를 배운다는 뜻입니다.

### 왜 '확률'로 접근해야 할까

- **정답이 하나가 아님**: 숫자 '7'을 쓴 손글씨는 사람마다 모양, 굵기, 기울기가 전부 다릅니다.
- **이미지는 분포다**: 세상의 모든 고양이 사진은 색, 자세, 배경이 달라도 "고양이답다"는 공통 패턴 위에 흩어져 존재합니다.

그래서 생성형 모델의 목표는 이렇게 정리됩니다.

> **"데이터가 만들어졌을 가능성 공간(잠재 공간, Latent Space)을 학습하자."**
> 이 공간에서 점 하나를 뽑으면(샘플링), 그것이 곧 새로운 이미지가 된다.

![왼쪽에는 '왜 확률인가'에 대한 두 가지 이유(정답이 하나가 아님, 이미지는 분포다)와 '가능성 공간을 학습하자'는 목표 박스가 있고, 오른쪽에는 손글씨 숫자 '3' 이미지 세 장이 매핑을 거쳐 2차원 잠재 공간의 점들로 흩어지고 그 사이에서 새 샘플을 뽑는 latent space 개념도가 있는 슬라이드](/assets/img/posts/generative-ai/02_probability_and_latent_space.svg)

*비슷한 이미지는 잠재 공간에서 가까이 모이고, 점과 점 사이를 샘플링하면 그 중간 성격의 새 이미지가 나온다*

이미지 생성 AI를 구현하는 대표적인 접근은 세 갈래입니다.

| 접근 | 대표 모델 | 한 줄 요약 |
|---|---|---|
| 경쟁적 접근 | **GAN** | 두 신경망이 서로 경쟁하며 학습 |
| 구조적 접근 | **DCGAN** | GAN에 합성곱 구조를 도입해 안정화 |
| 확률적 접근 | **VAE** | 데이터를 확률 분포로 직접 모델링 |

지금부터 이 셋을 순서대로 살펴봅니다.

---

## 3. GAN — 경쟁을 통한 학습 {#gan}

### 3.1 기본 개념

> **GAN (Generative Adversarial Network)**
> 두 개의 신경망(생성자와 판별자)이 서로 경쟁(Adversarial)하며 학습하여 진짜 같은 데이터를 생성하는 모델. 2014년 처음 제안되었습니다.

GAN을 이해하는 가장 쉬운 비유는 **위조지폐범과 경찰**입니다.

- **생성자(Generator, G)**: 위조지폐범. "경찰을 속일 만한 위조지폐를 만든다."
- **판별자(Discriminator, D)**: 경찰. "진짜 돈과 위조지폐를 구분한다."

이 둘이 서로 맞서서 경쟁하는 과정에서 위조지폐(생성된 이미지)의 품질이 점점 정교해집니다.

### 3.2 적대적 학습 루프 (Adversarial Training Loop)

1. **Random Noise (잠재 벡터 z)**: 무작위 노이즈 벡터에서 시작합니다.
2. **Generator(G)**: 이 노이즈를 입력받아 가짜 이미지 $G(z)$를 생성합니다.
3. **Discriminator(D)**: 가짜 이미지와 실제 이미지(Training Data)를 함께 입력받아 진짜(1)/가짜(0)를 판별합니다.
4. 이 과정이 반복되며, G는 더 진짜 같은 이미지를, D는 더 정확한 판별 능력을 갖도록 학습됩니다.

![GAN의 정의, Random Noise(z)가 Generator를 거쳐 Fake Image가 되고 Real Images와 함께 Discriminator에 입력되어 Real(1)/Fake(0)를 예측하는 Adversarial Training Loop, 그리고 Key Players(Generator·Discriminator)와 Minimax Game 설명을 함께 담은 슬라이드](/assets/img/posts/generative-ai/03_gan_concept_and_loop.svg)

*G는 D를 속이는 방향으로, D는 G에게 속지 않는 방향으로 — 같은 신호를 두고 반대로 학습한다*

### 3.3 미니맥스 게임 (Minimax Game)

GAN의 학습은 게임 이론의 **제로섬 게임(Zero-Sum Game)** 과 유사합니다.

- **Generator**: 판별자가 실수하도록 유도 (D의 성공을 최소화)
- **Discriminator**: 진짜와 가짜를 정확히 분류 (자신의 정확도를 최대화)

이론적으로 학습이 완벽히 진행되면, 판별자가 진짜와 가짜를 구별할 확률이 50%(완전히 헷갈리는 상태)에 도달하는 **내시 균형(Nash Equilibrium)** 에 이르며, 이때 Generator가 만드는 이미지는 진짜와 구별할 수 없을 만큼 정교해집니다.

![미니맥스 게임 정의(Generator는 D의 성공을 최소화, Discriminator는 자신의 정확도를 최대화)와, 실제 학습 과정에서 Generator Loss·Discriminator Loss가 뚜렷한 수렴 없이 계속 진동하는 모습을 보여주는 그래프, 그리고 하단에 불안정한 학습과 모드 붕괴 문제가 함께 담긴 슬라이드](/assets/img/posts/generative-ai/05_gan_minimax_and_loss.svg)

*이론적으로는 내시 균형에 수렴해야 하지만, 실제 학습 곡선은 G와 D가 서로 밀고 당기며 계속 진동하는 모습을 보인다*

### 3.4 GAN의 수학적 정의

판별자와 생성자의 손실 함수는 다음과 같습니다.

$$\mathcal{L}_D = -\big[\,\log D(x) + \log(1 - D(G(z)))\,\big]$$

$$\mathcal{L}_G = -\log D(G(z))$$

- $D(x)$: 진짜 이미지 $x$에 대한 판별자의 출력 (진짜일 확률)
- $D(G(z))$: 가짜 이미지에 대한 판별자의 출력

$\mathcal{L}_D$와 $\mathcal{L}_G$를 합쳐 하나의 게임으로 표현하면 **미니맥스 목적 함수**가 됩니다.

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log(1 - D(G(z)))\big]$$

- 판별자(D)는 이 값을 **최대화**하려 하고,
- 생성자(G)는 이 값을 **최소화**하려 합니다.

> **참고**: $\mathcal{L}_D$는 $V(D,G)$에 $-1$을 곱한 것과 정확히 같아서 그대로 유도됩니다. 반면 원래 미니맥스식대로라면 $G$는 $\log(1-D(G(z)))$를 최소화해야 하는데, 학습 초반 $D$가 너무 강할 때 이 항의 기울기가 거의 0에 가까워지는 문제가 있습니다. 그래서 실무에서는 $G$가 대신 $\log D(G(z))$를 최대화하도록 바꾼 **non-saturating loss**($\mathcal{L}_G = -\log D(G(z))$)를 씁니다. 즉 $\mathcal{L}_D$는 $V(D,G)$의 직접적인 유도지만, $\mathcal{L}_G$는 학습 안정성을 위한 실전 대체식이라는 점이 다릅니다.

하나의 식 안에 정반대 목표가 공존한다는 점이 GAN 학습이 본질적으로 "경쟁"일 수밖에 없는 이유입니다.

![입력 노이즈 벡터 z가 생성기를 거쳐 생성된(가짜) 이미지가 되고, 실제 이미지와 함께 판별기에 입력되어 진짜/가짜를 판별하는 구조도. 판별기의 출력에서 판별기 손실과 생성기 손실이 각각 점선 화살표로 되먹임되는 모습](/assets/img/posts/generative-ai/04_gan_structure_diagram.svg)

*판별기 손실은 판별기 자신을, 판별기를 속인 정도(생성기 손실)는 생성기를 업데이트하는 신호로 쓰인다 — $\mathcal{L}_D$와 $\mathcal{L}_G$가 실제로 흐르는 경로*

### 3.5 GAN 학습의 현실적인 문제

이론적으로는 우아하지만, 실제 GAN 학습에는 잘 알려진 두 가지 고질적 문제가 있습니다.

- **불안정한 학습(Instability)**: Loss가 수렴하지 않고 계속 진동하는 현상. G와 D가 서로 경쟁하는 구조라 균형점을 찾기 어렵습니다.
- **모드 붕괴(Mode Collapse)**: Generator가 다양성을 잃고, 판별자를 속이기 쉬운 특정 패턴만 반복 생성하는 현상. 예를 들어 손글씨 전체를 배워야 하는데 '1'만 그럴듯하게 계속 만들어냅니다.

이런 문제를 구조적으로 개선하기 위해 등장한 것이 **DCGAN**입니다.

---

## 4. DCGAN — 구조로 안정성을 잡다 {#dcgan}

### 4.1 DCGAN이 바꾼 것

초기 GAN은 이미지의 공간적 구조를 고려하지 않고 **완전연결층(Fully Connected Layer)** 만으로 생성자와 판별자를 구성했습니다. 그 결과 학습이 불안정하고 고해상도 이미지 생성은 거의 작동하지 않았습니다.

**DCGAN(Deep Convolutional GAN)** 은 아래와 같은 구조 설계 가이드라인을 제시해 이 문제를 크게 개선했습니다.

- 완전연결층 대신 **합성곱(Convolution)** 구조 사용
- 풀링(Pooling) 대신 **strided convolution** 으로 다운/업샘플링
- **배치 정규화(Batch Normalization)** 로 학습 안정화
- 생성자는 ReLU(출력층만 Tanh), 판별자는 LeakyReLU 사용

즉 DCGAN은 "합성곱을 썼다"는 한 가지 변화가 아니라, **생성자·판별자 전체를 CNN 기반으로 공간적으로 재설계한 형태**입니다.

### 4.2 DCGAN 생성자(Generator) 구조

생성자는 아래 흐름으로 작은 벡터를 점점 큰 이미지로 확장해갑니다.

```
64차원 노이즈 벡터 z
  → 선형 계층 (Fully Connected)  : 64 → 16×16×128  (= 32,768개 값 계산)
  → reshape                      : 16×16×128 특징 맵으로 재배열
  → 업샘플링 + 합성곱 1           : 32×32×128
  → 업샘플링 + 합성곱 2           : 64×64×64
  → 합성곱 3 (출력)              : 64×64×3   (RGB 이미지)
```

여기서 네 가지를 명확히 구분해서 이해하는 것이 중요합니다.

1. **선형 계층은 "계산"이다.** 64개의 입력값을 서로 다른 32,768가지 가중치 조합으로 섞어(가중합 + 편향), 완전히 새로운 숫자 32,768개를 만들어냅니다.
2. **reshape는 "계산이 아니라 재배열"이다.** 방금 계산된 32,768개의 숫자를 추가 연산 없이 그대로 16×16×128 모양의 3차원 상자에 순서대로 담아 붙입니다.
3. **업샘플링은 "복제"다.** Nearest Neighbor 방식으로, 픽셀 1개가 2×2로 그대로 복제되어 해상도가 2배로 커집니다.
4. **업샘플링 직후의 합성곱**이 이 뭉텅뭉텅하게 복제된 픽셀 블록을 자연스럽게 다듬어줍니다.

여기서 `64`(잠재 벡터 차원), `16×16×128`(시작 특징 맵 크기) 같은 숫자들은 계산으로 저절로 나오는 값이 아니라, **설계자가 미리 정한 하이퍼파라미터**라는 점도 기억해두면 좋습니다.

![64차원 노이즈 벡터 z가 선형 계층과 reshape를 거쳐 16×16×128 특징 맵이 되고, 업샘플링+합성곱을 두 번 거치며 32×32×128, 64×64×128을 지나 합성곱 계층에서 64×64×64로, 마지막에 64×64×3 RGB 이미지로 확장되는 DCGAN 생성기 아키텍처 다이어그램](/assets/img/posts/generative-ai/06_dcgan_generator_architecture.svg)

*생성자는 작은 벡터에서 출발해 공간 크기는 키우고(16→32→64) 채널 수는 줄이며(128→64→3) 이미지를 키워간다*

이 흐름을 PyTorch `nn.Module`로 옮기면 다음과 같습니다.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 16 * 16 * 128)  # 64 → 32,768 (계산)
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),                          # 16×16 → 32×32 (복제)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),                          # 32×32 → 64×64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),                                            # [-1, 1] 범위 출력
        )

    def forward(self, z):
        x = self.fc(z)                 # (B, 64) → (B, 32768)
        x = x.view(-1, 128, 16, 16)     # reshape (재배열, 계산 아님)
        return self.conv_blocks(x)      # (B, 3, 64, 64)
```

업샘플링이 실제로 어떻게 픽셀을 복제하는지는, 아래 위젯에서 직접 눌러보면서 확인하면 훨씬 이해가 빠릅니다. 4×4 격자가 8×8로 업샘플링되는 과정을 체험해볼 수 있습니다.

<div style="border: 1px solid #d3d1c7; border-radius: 12px; overflow: hidden; margin: 24px 0;">
  <iframe src="/assets/html/generative_ai/dcgan_generator_upsampling_demo.html" width="100%" height="500" style="border: none; display: block;" title="DCGAN 생성자 업샘플링 시각화"></iframe>
</div>

### 4.3 DCGAN 판별자(Discriminator) 구조

판별자는 생성자와 정반대 방향으로, 이미지를 점점 작고 추상적인 벡터로 압축해갑니다.

```
64×64×3 RGB 이미지
  → 합성곱 1 (stride 2)  : 32×32×16
  → 합성곱 2 (stride 2)  : 16×16×32
  → 합성곱 3 (stride 2)  : 8×8×64
  → 합성곱 4 (stride 2)  : 4×4×128
  → Flatten + 완전연결층 + Sigmoid
  → 확률값 하나 (0~1, 진짜/가짜)
```

stride=2인 합성곱을 사용해 별도의 풀링 층 없이도 공간 크기를 절반씩 줄이면서, 채널 수는 점점 늘려갑니다. 결국 판별자는 우리가 흔히 아는 **이진 분류(Binary Classification) 모델** 과 본질적으로 같은 구조입니다.

![64×64×3 RGB 이미지가 stride=2인 합성곱 4개(채널 3→16→32→64→128, 공간 크기 64→32→16→8→4)를 거쳐 압축되고, Flatten과 완전연결층·Sigmoid를 지나 0~1 사이의 진짜/가짜 확률 하나로 요약되는 DCGAN 판별기 아키텍처 다이어그램](/assets/img/posts/generative-ai/07_dcgan_discriminator_architecture.svg)

*판별기는 생성기와 정반대로, 공간은 줄이고(64→32→16→8→4) 채널은 늘리며(3→16→32→64→128) 이미지를 압축해간다*

이 흐름을 PyTorch `nn.Module`로 옮기면 다음과 같습니다.

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),                       # 64×64 → 32×32

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),                       # 32×32 → 16×16

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),                       # 16×16 → 8×8

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),                       # 8×8 → 4×4
        )
        self.fc = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv_blocks(x)          # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)        # Flatten → (B, 2048)
        return torch.sigmoid(self.fc(x)) # (B, 1) 진짜/가짜 확률
```

### 4.4 생성자 vs 판별자 — 거울 구조

| | 생성자 | 판별자 |
|---|---|---|
| 입력 | 64차원 노이즈 벡터 | 64×64×3 이미지 |
| 방향 | 압축 → 확장 (디코더 역할) | 이미지 → 압축 (인코더 역할) |
| 공간 크기 | 증가 (16→32→64) | 감소 (64→32→16→8→4) |
| 채널 수 | 감소 (128→128→64→3) | 증가 (3→16→32→64→128) |

---

## 5. GAN의 확장: SRGAN, CycleGAN, Pix2Pix {#gan-variants}

GAN의 기본 골격(Generator–Discriminator, 미니맥스 손실)은 그대로 둔 채, **손실 함수·구조·입력 형태**를 목적에 맞게 바꾼 다양한 변형들이 등장했습니다.

- **SRGAN**: 저해상도 이미지를 고해상도로 변환 (초해상도)
- **CycleGAN**: 생성자 2개를 사용해, 짝이 맞지 않는 데이터로도 이미지 간 스타일 변환 수행 (예: 말 ↔ 얼룩말)
- **LSGAN**: 크로스엔트로피 대신 평균 제곱 오차(MSE)를 사용해 학습 안정성 향상
- **Pix2Pix**: 단순 이미지 생성이 아니라 이미지→이미지 변환(image-to-image translation)에 특화

### 5.1 Pix2Pix — U-Net 기반 생성자

Pix2Pix의 입력은 노이즈가 아니라 **이미지 그 자체**입니다. 그래서 구조는 DCGAN과 달리 "이미지를 압축했다가(Encoder) 다시 복원하는(Decoder)" **U-Net** 형태를 가집니다.

- **Encoder (Downsampling)**: 여러 단계의 합성곱으로 이미지를 점진적으로 압축 (합성곱 → 정규화 → LeakyReLU)
- **Decoder (Upsampling)**: 업샘플링 + 합성곱으로 다시 이미지를 복원 (ReLU, 마지막 단계는 Tanh)
- **Skip Connection**: Encoder의 각 단계 출력을 Decoder의 대응 단계에 직접 이어붙여(concat), 압축 과정에서 잃기 쉬운 세밀한 디테일(선, 경계, 질감)을 보존

![입력 이미지가 Encoder(합성곱→정규화→LeakyReLU)를 거쳐 점점 좁게 압축되고, Decoder(업샘플링→정규화→ReLU, 마지막은 Tanh)를 거쳐 다시 넓게 복원되며 Translated Image로 출력되는 U자 구조. 같은 높이의 Encoder-Decoder 단계 사이를 점선 화살표(skip connection)가 가로질러 연결하는 Pix2Pix Generator 구조도](/assets/img/posts/generative-ai/08_pix2pix_unet_generator.svg)

*U-Net은 압축부에서 잃은 고주파 디테일을 skip connection으로 복원부에 그대로 전달한다*

### 5.2 Pix2Pix — PatchGAN 판별자

Pix2Pix의 판별자는 원본 이미지와 생성된 이미지를 채널 방향으로 이어붙여(concat) 함께 입력받습니다. 그리고 결정적인 차이가 있습니다.

- 일반 GAN 판별자는 이미지 전체를 보고 진짜/가짜를 **하나의 숫자**로 판별합니다.
- **PatchGAN**은 이미지를 70×70 크기의 작은 패치(patch) 단위로 나눠, **각 패치별로** 진짜/가짜를 판별하고 평균을 냅니다.

이렇게 국소 단위로 꼼꼼히 판별하면, 전체적인 형태보다 **세밀한 질감(realistic texture)** 을 훨씬 민감하게 잡아낼 수 있습니다.

![Real Image와 Translated Image가 채널 방향으로 concat되어 conv layer 4개(채널 6→16→64→128→256)를 거치고, flatten과 완전연결층을 지나 '이미지 쌍이 진짜/가짜일 확률' 하나로 출력되는 PatchGAN 판별기 구조도. 하단에는 패치 단위로 판별하는 이유가 함께 설명되어 있다](/assets/img/posts/generative-ai/09_patchgan_discriminator.svg)

*이미지 전체를 하나의 숫자로 판별하는 대신, 70×70 패치마다 진짜/가짜를 판별해 평균 내는 것이 PatchGAN의 핵심이다*

---

## 6. VAE — 확률로 접근하는 생성 {#vae}

### 6.1 기본 개념

> **VAE (Variational AutoEncoder)**
> 이미지를 잠재 공간(latent space)의 확률 분포로 변환하여 새로운 데이터를 생성하는 모델.

GAN이 "경쟁"을 통해 간접적으로 이미지 생성법을 배웠다면, VAE는 **오토인코더(AutoEncoder)** 구조를 기반으로 이미지를 확률 분포로 직접 표현하고, 그 분포에서 샘플링해 이미지를 생성합니다.

**개념 비유**
- **Encoder**: 사진을 보고 상세한 설명서(특징)를 작성하는 사람
- **Decoder**: 그 설명서만 보고 다시 그림을 그리는 사람

![Input Image가 Encoder를 거쳐 잠재 공간의 평균 μ와 표준편차 σ로 압축되고, 그 분포에서 z ~ N(μ,σ)를 샘플링한 뒤 Decoder를 통과해 Reconstructed(New) Image로 복원되는 좌우 흐름도. 우측에는 Encoder를 '사진 보고 설명서를 쓰는 사람', Decoder를 '설명서만 보고 그림을 그리는 사람'에 비유한 개념 박스가 함께 있다](/assets/img/posts/generative-ai/10_vae_structure_flow.svg)

*Encoder는 이미지를 하나의 점이 아니라 (μ, σ)라는 분포로 인코딩하고, 그 분포에서 뽑은 z를 Decoder가 이미지로 되돌린다*

### 6.2 일반 AutoEncoder(AE)와의 차이

| | AutoEncoder (AE) | VAE |
|---|---|---|
| 잠재 공간 | 고정된 하나의 점 | 확률 분포 $(\mu, \sigma)$ |
| 목적 | 단순 압축 및 복원 | 압축·복원 + **생성** |
| 새 데이터 생성 | 어려움 (빈 공간의 의미가 보장되지 않음) | 샘플링으로 가능 |

AE는 이미지를 잠재 공간의 고정된 점 하나로 압축합니다. 점과 점 사이의 빈 공간은 학습되지 않았기 때문에, 그 사이에서 샘플링하면 망가진 이미지가 나오기 쉽습니다.

VAE는 이미지를 점이 아니라 **분포(구름)** 로 인코딩하고, 이 분포들이 잠재 공간 전체에 걸쳐 빈틈없이 이어지도록 학습됩니다. 그 결과 잠재 공간 아무 데서나 샘플링해도 그럴듯한 이미지가 나옵니다.

![AutoEncoder(AE)는 잠재 공간이 고정값이라 새로운 생성(Sampling)이 어렵고, VAE는 잠재 공간이 확률 분포라 데이터의 특성 분포를 학습해 Sampling으로 생성이 가능하다는 것을 나란히 비교한 카드형 다이어그램](/assets/img/posts/generative-ai/11_ae_vs_vae_comparison.svg)

*AE의 잠재 공간은 점 사이가 비어 있고, VAE는 KL 항 덕분에 분포들이 겹치며 연속적으로 이어진다*

### 6.3 재매개변수화 트릭 (Reparameterization Trick)

VAE 학습에서 반드시 알아야 할 핵심 트릭입니다.

- Encoder는 입력 이미지를 **평균 $\mu$와 표준편차 $\sigma$** 로 압축합니다.
- 문제는, "$z \sim \mathcal{N}(\mu, \sigma)$에서 직접 샘플링"하는 연산은 **미분이 불가능**해서 역전파가 안 된다는 점입니다.
- 그래서 무작위성을 별도의 변수 $\varepsilon$(표준정규분포에서 뽑은 작은 노이즈)로 분리합니다.

$$z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

이렇게 하면 $z$는 여전히 원하는 분포를 따르면서도, $\mu$와 $\sigma$에 대한 역전파가 정상적으로 가능해집니다.

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)     # logvar를 학습하고 std로 변환 (수치 안정성)
    eps = torch.randn_like(std)       # N(0, 1)에서 뽑은 노이즈
    return mu + eps * std
```

### 6.4 VAE의 손실 함수

VAE는 두 가지 손실을 동시에 최소화합니다.

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}\big[-\log p(x \mid z)\big]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}\big(q(z \mid x)\,\|\,\mathcal{N}(0, I)\big)}_{\text{KL Divergence}}$$

- **Reconstruction Loss (재구성 손실, 보통 BCE)**: Decoder가 복원한 이미지가 원본과 얼마나 비슷한지 측정
- **KL Divergence**: Encoder가 만든 분포 $(\mu, \sigma)$가 표준정규분포 $\mathcal{N}(0, I)$에서 너무 벗어나지 않도록 정규화 → 이것이 잠재 공간을 "빈틈없이 이어지도록" 만드는 핵심 장치

두 손실을 동시에 최소화하면서, "원본을 잘 복원하면서도 잠재 공간을 깔끔하게 정리"하는 두 목표를 함께 달성합니다.

Decoder의 출력층은 보통 **Sigmoid** 를 사용해 $[0, 1]$ 범위의 이미지를 출력합니다 (DCGAN 생성자가 Tanh로 $[-1, 1]$을 출력하는 것과 대비됩니다).

### 6.5 VAE로 이미지 생성하기

학습이 끝난 뒤 새 이미지를 생성할 때는 흥미롭게도 **Encoder를 쓰지 않습니다.**

| 단계 | 학습(Training) 때 | 생성(Generation) 때 |
|---|---|---|
| Encoder | 사용함 (이미지 → μ, σ) | **사용 안 함** |
| z | $\mu + \sigma \odot \varepsilon$ (재매개변수화) | $\mathcal{N}(0, I)$에서 직접 샘플링 |
| Decoder | 사용함 | 사용함 |

```python
z = torch.randn(64, latent_dim)   # N(0, I)에서 그냥 뽑는다
samples = decoder(z)              # Encoder 없이 바로 이미지 생성
```

KL Divergence 손실 덕분에 잠재 공간 전체가 표준정규분포와 비슷하게 정리되도록 학습되었기 때문에, 그냥 $\mathcal{N}(0, I)$에서 무작위로 뽑은 $z$를 Decoder에 넣기만 해도 그럴듯한 이미지가 만들어집니다.

![torch.randn(64, 20)으로 뽑은 Random Noise(z)가 Decoder 하나만 거쳐, MNIST 숫자들이 격자로 나열된 Generated Samples로 바로 출력되는 흐름도. Encoder 없이 Decoder만으로 생성이 이루어진다는 점이 강조되어 있다](/assets/img/posts/generative-ai/12_vae_generation_sampling.svg)

*Encoder는 학습에만 쓰이고, 생성 단계에서는 무작위 노이즈와 Decoder만으로 새 이미지를 만들어낸다*

---

## 7. GAN vs VAE 최종 비교 {#gan-vs-vae}

| 기준 | VAE | GAN |
|---|---|---|
| 학습 안정성 | 명확한 Loss로 수렴이 안정적 | 균형이 깨지면 학습 실패 (Mode Collapse) |
| 이미지 품질 | 평균값을 학습해 다소 흐릿함 | 디테일하고 선명한 이미지 생성 |
| 수학적 기반 | 확률 분포 근사 (Likelihood) | 게임 이론 & 내시 균형 |
| 구현 난이도 | 중간 | 높음 (하이퍼파라미터에 매우 민감) |
| 잠재 공간 | 연속적이고 해석·조작하기 쉬움 | 상대적으로 다루기 어려움 |

두 모델 모두 "결국 데이터가 존재하는 분포를 배운다"는 목적지는 같지만, 그 방식이 **경쟁(GAN)** 이냐 **확률 근사(VAE)** 냐에 따라 학습 안정성, 이미지 품질, 잠재 공간의 성격이 정반대의 트레이드오프를 보입니다.

**언제 무엇을 쓸까?**

- 잠재 공간을 보간·편집하며 제어하고 싶거나, 이상 탐지·압축이 목적이거나, 학습 안정성이 중요하다 → **VAE**
- 최대한 선명하고 사실적인 이미지가 필요하다 → **GAN**
- 스타일 변환, 이미지→이미지 변환이 목적이다 → **GAN 계열 (Pix2Pix, CycleGAN)**

실무에서는 종종 둘을 결합해서 씁니다. 대표적으로 **Stable Diffusion** 은 VAE로 이미지를 작은 잠재 공간으로 압축한 뒤, 그 위에서 diffusion 과정을 수행하고 다시 VAE Decoder로 복원하는 방식으로, VAE의 압축 능력을 diffusion 모델의 계산 효율화에 활용합니다.

---

## 8. 마무리 {#summary}

생성형 이미지 AI를 관통하는 단 하나의 아이디어는 다음 문장으로 요약할 수 있습니다.

> **생성형 모델은 데이터의 "분포"를 학습한다. 단순 암기가 아니라, 데이터가 생성될 확률 공간을 이해하고 그 공간에서 새로운 샘플을 만들어낸다.**

GAN은 이 분포를 **경쟁**을 통해 암묵적으로 흉내 내는 법을 배우고, VAE는 이 분포를 **$\mu, \sigma$라는 확률 파라미터**로 명시적으로 학습합니다. 접근 방식은 다르지만 목적지는 같다는 점이 이 두 모델을 이해하는 핵심입니다.

![생성형 모델은 '분포(Distribution)'를 학습한다는 요약 박스 아래, VAE(잠재 공간의 확률적 분포를 가정하고 샘플링하여 생성)와 GAN(생성자와 판별자의 경쟁적 학습으로 사실적 이미지 생성) 두 카드, 그리고 활용 분야로 스타일 전이·이미지 생성/복원·데이터 증강 세 아이콘 카드를 나란히 보여주는 마무리 요약 슬라이드](/assets/img/posts/generative-ai/13_summary_wrapup.svg)

**핵심 체크리스트**

- [ ] 판별형은 $p(y \mid x)$, 생성형은 $p(x)$를 학습한다
- [ ] GAN은 Generator와 Discriminator의 미니맥스 게임이다
- [ ] GAN의 고질병은 학습 불안정과 Mode Collapse다
- [ ] DCGAN은 생성자·판별자를 CNN으로 재설계해 안정성을 높였다
- [ ] 생성자에서 선형 계층은 "계산", reshape는 "재배열", 업샘플링은 "복제"다
- [ ] Pix2Pix는 U-Net 생성자 + PatchGAN 판별자로 이미지→이미지 변환에 특화된다
- [ ] VAE는 이미지를 점이 아니라 분포로 인코딩하고, KL 항이 잠재 공간을 연속적으로 만든다
- [ ] 재매개변수화 트릭 $z = \mu + \sigma \odot \varepsilon$ 덕분에 샘플링을 통과하는 역전파가 가능하다
- [ ] VAE 생성 단계에서는 Encoder를 쓰지 않고 $\mathcal{N}(0, I)$에서 직접 샘플링한다

**활용 분야**

- **스타일 전이 (Style Transfer)**: 사진을 다른 화풍으로, 낮 사진을 밤 사진으로
- **이미지 복원 (Inpainting)**: 손상되거나 가려진 영역을 자연스럽게 채워 넣기
- **데이터 증강 (Augmentation)**: 부족한 학습 데이터를 생성 모델로 보충

---

*이 글은 생성형 이미지 AI(GAN, DCGAN, Pix2Pix, VAE)의 원리를 개념부터 구조와 수식까지 정리한 학습 노트입니다.*
