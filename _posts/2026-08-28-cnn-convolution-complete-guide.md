---
title: "CNN 완전 정복: 이론부터 PyTorch 구현까지"
date: 2026-08-28 04:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, cnn, convolution, pooling, cifar-10, image-classification, machine-learning]
description: "MLP가 왜 이미지에 약한지부터 시작해 합성곱·스트라이드·패딩·풀링 연산의 원리, Feature Extractor와 Classifier의 역할 분리, 하이퍼파라미터 설계 가이드, 그리고 CIFAR-10을 분류하는 SimpleCNN을 PyTorch로 구현하는 전체 흐름을 정리했다."
---

> MLP는 왜 이미지를 잘 못 다룰까요? 합성곱과 풀링은 어떻게 이 문제를 해결할까요? 그리고 실제로 PyTorch로 CIFAR-10 이미지를 분류하는 CNN은 어떻게 만들까요? 이 글 하나로 CNN의 등장 배경부터 실제 구현까지 전체 흐름을 정리합니다.
>
> [지난 글]({% post_url 2026-08-20-pytorch-mnist-multiclass-classification %})에서는 손글씨 숫자(MNIST)를 완전결합 신경망으로 분류했습니다. 이번 글은 한 단계 더 들어가, 합성곱이라는 연산이 어떻게 이미지의 공간 구조를 살리는지를 다룹니다.
>
> CNN의 큰 그림과 실전 비교(완전결합 신경망 vs CNN)는 [CNN 기반 이미지 분류, 제대로 이해하기]({% post_url 2026-08-20-cnn-image-classification %})에서 먼저 다뤘습니다. 이 글은 그 위에서 합성곱 연산의 산술, 활성화·풀링의 선택지, 하이퍼파라미터 설계 결정을 레퍼런스처럼 깊게 파고듭니다.

## 📌 목차

1. [CNN은 왜 등장했는가](#why-cnn)
2. [합성곱 연산의 원리](#convolution)
3. [Filter, Stride, Padding](#filter-stride-padding)
4. [특징맵(Feature Map)](#feature-map)
5. [활성화 함수: ReLU](#relu)
6. [Pooling](#pooling)
7. [CNN 전체 구조](#cnn-structure)
8. [Feature Extractor vs Classifier](#extractor-vs-classifier)
9. [하이퍼파라미터 설계 가이드](#hyperparameter-guide)
10. [실전: CIFAR-10으로 CNN 학습시키기](#cifar10-practice)
11. [모델 평가와 시각화](#evaluation)
12. [마무리](#summary)

---

## 1. CNN은 왜 등장했는가 {#why-cnn}

이미지를 다루는 딥러닝 모델이라고 하면 자연스럽게 CNN을 떠올리지만, 애초에 기존의 MLP(다층 퍼셉트론)로는 왜 이미지를 제대로 처리할 수 없었는지부터 짚고 넘어가야 합니다.

### 기존 MLP의 근본적 한계

**문제 1. 공간 정보 손실**

이미지를 MLP에 입력하려면 2차원(또는 3차원) 데이터를 1차원 벡터로 펼쳐야(flatten) 합니다. 이 과정에서 픽셀 간 공간적 관계가 완전히 사라집니다. 예를 들어 32×32 RGB 이미지는 3,072개의 독립적인 입력 뉴런으로 처리되는데, 바로 옆에 붙어 있던 픽셀이라는 정보 자체가 의미를 잃어버립니다.

**문제 2. 파라미터 폭발**

256×256×3(RGB) 이미지를 1,000개의 은닉 뉴런과 완전 연결(fully connected)하면 약 1억 9,600만 개(256×256×3×1,000)의 파라미터가 필요합니다. 이렇게 파라미터가 많아지면 과적합 위험이 극도로 커지고, 학습 자체가 현실적으로 어려워집니다.

**문제 3. 위치 불변성 부족**

같은 객체라도 이미지 안 위치가 달라지면 MLP는 완전히 다른 입력으로 인식합니다. 즉 물체가 평행이동하거나 회전하면 제대로 인식하지 못하는 구조적 약점을 갖습니다.

![MLP는 이미지를 1D 벡터로 펼치면서 공간 정보를 잃고, 파라미터가 폭발하며, 위치가 바뀌면 다르게 인식한다](/assets/img/posts/cnn-convolution-complete-guide/01_mlp_limitation.png)

### CNN이 이 문제를 해결하는 방식

CNN은 다음 세 가지 핵심 아이디어로 위 문제들을 극복합니다.

- **Local Connectivity (지역 연결성)**: 전체 이미지가 아니라 작은 영역만 보고 지역적 패턴을 인식 → 공간 정보 보존
- **Parameter Sharing (파라미터 공유)**: 하나의 필터를 이미지 전체에 재사용 → 파라미터 수를 획기적으로 감소
- **Spatial Hierarchy (공간적 계층 구조)**: 층이 깊어질수록 저수준 특징(엣지 등)에서 고수준 특징(형태, 객체 등)으로 계층적으로 학습

---

## 2. 합성곱 연산의 원리 {#convolution}

CNN의 이름이기도 한 **합성곱(Convolution)**은, 입력 이미지와 필터(커널)의 element-wise 곱셈을 수행한 뒤 합산하는 연산입니다. 필터가 슬라이딩 윈도우 방식으로 전체 이미지를 순회하며, 각 위치에서 특정 패턴의 존재 여부를 검출합니다.

### 수학적 정의

$$Output(i,j) = \sum_m \sum_n Input(i+m, j+n) \times Kernel(m,n)$$

풀어서 말하면, 출력의 (i, j) 위치 값은 입력 이미지 해당 위치 주변 영역과 커널 값을 각각 곱한 뒤 모두 더한 값입니다.

![합성곱 연산 과정: 입력의 3×3 영역과 커널을 요소별로 곱한 뒤 모두 더해 출력 한 칸을 만든다](/assets/img/posts/cnn-convolution-complete-guide/02_convolution_operation.png)

이 연산을 통해 엣지, 코너, 텍스처 같은 저수준 특징을 자동으로 검출할 수 있으며, 필터의 가중치는 역전파 학습을 통해 최적화됩니다.

### 예시: Edge Detection

대표적인 3×3 수평 엣지 검출 필터 예시를 보겠습니다.

```
[-1 -1 -1]
[ 0  0  0]
[ 1  1  1]
```

위쪽 행은 음수, 아래쪽 행은 양수로 구성되어 있습니다. 이 필터가 이미지 위를 지나갈 때 위아래 밝기 차이가 큰 부분(수평 방향 경계선)에서 큰 값이 출력됩니다. 즉 합성곱 출력값이 클수록 그 위치에 해당 패턴(수평 경계)이 강하게 존재한다는 뜻입니다.

### 합성곱의 가장 큰 장점: Translation Invariance

같은 필터 하나가 이미지 전체를 동일하게 순회하기 때문에, 특정 패턴이 이미지의 어느 위치에 있어도 동일하게 검출할 수 있습니다. 이것이 앞서 언급한 MLP의 "위치 불변성 부족" 문제를 CNN이 해결하는 핵심 원리입니다.

---

## 3. Filter, Stride, Padding {#filter-stride-padding}

합성곱의 출력 크기를 결정하는 세 가지 하이퍼파라미터를 알아보겠습니다.

### 필터 (Filter/Kernel)

- 일반적으로 3×3, 5×5, 7×7 크기를 사용합니다.
- 필터의 깊이는 입력 채널 수와 동일하며, 필터 개수는 출력 Feature Map의 채널 수를 결정합니다.
- VGG, ResNet 등 현대적 아키텍처는 주로 3×3 필터를 여러 겹 쌓는 방식을 선호합니다. 3×3을 두 번 쌓으면 5×5 한 번과 수용 영역은 같으면서 파라미터는 더 적고(2×9 vs 25), 그 사이에 비선형 활성화가 한 번 더 들어가 표현력이 높아지기 때문입니다.

### 스트라이드 (Stride)

필터가 이동하는 간격을 제어합니다. stride=1은 한 칸씩, stride=2는 두 칸씩 이동하여 출력 크기를 1/2로 축소합니다. 큰 stride는 계산량을 줄여주지만 정보 손실이 발생할 수 있습니다. 최근 아키텍처 중에는 풀링 대신 stride=2 합성곱으로 다운샘플링을 처리하는 경우도 많습니다.

### 패딩 (Padding)

입력 이미지 주변에 0을 추가하여 출력 크기를 제어합니다.

- **Valid Padding**: 패딩 없이 크기가 감소
- **Same Padding**: 출력 크기를 입력과 동일하게 유지

### 출력 크기 계산 공식

$$Output = \frac{Input - Kernel + 2 \times Padding}{Stride} + 1$$

**실제 예시:** Input 32×32, Kernel 5×5, Stride 1, Padding 0 → Output = (32-5)/1+1 = **28×28**

### PyTorch 구현

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=5,
    stride=1,
    padding=2
)
# 출력: 32×32×64  (padding=2로 Same Padding 효과)
```

---

## 4. 특징맵(Feature Map) {#feature-map}

합성곱 연산의 출력을 **Feature Map**이라 부릅니다. 이는 입력 이미지에서 특정 패턴이 어디에 존재하는지를 나타내는 지도(map) 역할을 합니다. 필터 1개당 Feature Map 1개가 생성되며, 다중 필터를 사용하면 다양한 특징을 동시에 검출할 수 있습니다.

CIFAR-10 이미지를 예로 진행 과정을 따라가 보겠습니다.

| 단계 | 연산 | 결과 크기 |
|---|---|---|
| 입력 이미지 | - | 32×32×3 (RGB) |
| 첫 번째 합성곱 | 32 filters | 32×32×32 |
| 두 번째 합성곱 | 64 filters | 32×32×64 |
| 특징 추출 완료 | - | 64개 Feature Maps |

층을 거칠수록 채널 수(= Feature Map 개수)는 증가하면서, 공간 해상도는 줄고 각 채널이 담는 특징은 점점 더 추상적이고 복잡해집니다. 1장에서 말한 Spatial Hierarchy가 실제로 구현되는 지점입니다.

---

## 5. 활성화 함수: ReLU {#relu}

### ReLU 활성화 함수

$$ReLU(x) = max(0, x)$$

입력값이 양수면 그대로 통과시키고, 음수면 0으로 만들어버리는 단순한 함수입니다.

**ReLU가 CNN에서 가장 널리 쓰이는 이유:**

1. **계산이 매우 빠름**: 단순한 max 연산이라 지수 계산이 필요한 sigmoid/tanh보다 훨씬 빠릅니다.
2. **Gradient Vanishing(기울기 소실) 완화**: 양수 구간에서 기울기가 항상 1로 일정하게 유지되어, 층이 깊어져도 기울기가 잘 전달됩니다.
3. **희소성(Sparsity) 제공**: 음수 입력을 모두 0으로 만들어 일부 뉴런만 활성화되는 효율적인 표현을 만듭니다.

### 다른 활성화 함수들

- **LeakyReLU**: 음수 영역에서도 작은 기울기를 유지해 뉴런이 죽는 문제(Dying ReLU)를 완화
- **ELU**: 지수함수 기반의 부드러운 곡선
- **GELU**: Transformer 계열 모델에서 주로 사용

CNN은 합성곱(선형 연산) 다음에 반드시 비선형 활성화 함수를 적용해야 합니다. 그렇지 않으면 층을 아무리 깊게 쌓아도 결국 하나의 선형 변환과 다를 바 없어지기 때문입니다.

---

## 6. Pooling {#pooling}

### Pooling의 세 가지 효과

**1. 차원 축소**: 공간 해상도를 줄여 계산량을 크게 감소시킵니다. 2×2 풀링은 크기를 1/4로 줄입니다.

**2. 위치 불변성**: 객체가 미세하게 이동해도 안정적으로 같은 특징을 인식합니다. 객체의 정확한 위치보다 "존재 여부"가 중요해집니다.

**3. 수용 영역 확대**: 수용 영역(receptive field)이란 출력 한 픽셀이 영향을 받는 원본 이미지의 영역을 말합니다. 풀링을 거치면 다음 층의 뉴런이 더 넓은 원본 이미지 영역을 보게 되고, 그만큼 넓은 정보를 통합해 고수준 특징을 활성화합니다.

### Max Pooling vs Average Pooling

**Max Pooling**은 윈도우 내에서 최댓값을 선택하여 가장 두드러지는 특징을 보존합니다. 엣지나 코너 같은 강한 활성화를 유지하는 데 효과적입니다.

![Max Pooling 예시: 2×2 영역마다 최댓값만 남긴다](/assets/img/posts/cnn-convolution-complete-guide/03_max_pooling.png)

**Average Pooling**은 윈도우 내 평균값을 계산하여 전체적인 특징을 보존합니다. **Global Average Pooling(GAP)**은 분류기 직전에 사용되어 각 Feature Map을 하나의 값으로 압축하며, Fully Connected Layer 대신 사용해 파라미터 수를 크게 줄이는 데 활용됩니다.

```python
# Max Pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Global Average Pooling
gap = nn.AdaptiveAvgPool2d((1, 1))
```

> **주의:** 풀링 레이어는 학습 가능한 파라미터가 없습니다. 너무 많은 풀링은 정보 손실을 유발하므로 적절한 균형이 필요합니다.

---

## 7. CNN 전체 구조 {#cnn-structure}

지금까지 배운 요소들(합성곱, 활성화 함수, 풀링)이 실제로 어떻게 조합되어 전체 네트워크를 이루는지 살펴보겠습니다.

### 5단계 흐름

1. **Input Layer**: 32×32×3 RGB 이미지
2. **Convolutional Blocks**: `[CONV → ReLU → POOL]` × N 반복
3. **Flatten**: 다차원 텐서를 1D 벡터로 변환
4. **Fully Connected Layers**: `[FC → ReLU]` × M 반복
5. **Output Layer**: Softmax로 클래스 확률 출력

### LeNet-5 구조 예시

| 단계 | 연산 | 결과 크기 |
|---|---|---|
| Input | 32×32×1 (Grayscale) | - |
| Conv1 | 6 filters, 5×5 | 28×28×6 |
| Pool1 | Max 2×2 | 14×14×6 |
| Conv2 | 16 filters, 5×5 | 10×10×16 |
| Pool2 | Max 2×2 | 5×5×16 |
| Flatten | - | 400 features |
| FC1 | 400 → 120 | - |
| FC2 | 120 → 84 | - |
| FC3 | 84 → 10 classes | - |

LeNet-5의 채널 수는 6 → 16으로, 지금 기준으로는 작습니다. 이 모델은 1998년에 나온 역사적 사례이고, 아래 "채널 수를 2배씩 키운다"는 관례는 이후 AlexNet·VGG를 거치며 자리 잡은 것입니다.

### 설계 핵심 원칙

- **채널 수 증가**: 깊이가 깊어질수록 32 → 64 → 128 → 256으로 증가
- **공간 크기 감소**: 풀링을 통해 점진적으로 축소
- **계층적 학습**: Low-level → High-level 특징

CNN 설계에는 일관된 패턴이 있습니다. **"채널 수는 늘리고, 공간 크기는 줄이면서" 층을 깊게 쌓는 것**입니다. 이렇게 하면 계산량을 관리 가능한 수준으로 유지하면서도 점점 더 추상적인 특징을 학습할 수 있습니다.

---

## 8. Feature Extractor vs Classifier {#extractor-vs-classifier}

CNN 전체 구조는 크게 두 개의 기능적 블록으로 나눌 수 있습니다.

![CNN은 Conv+ReLU+Pool 블록으로 이루어진 Feature Extractor와, FC+Softmax로 이루어진 Classifier 두 블록으로 구성된다](/assets/img/posts/cnn-convolution-complete-guide/04_cnn_architecture.png)

| 구분 | Feature Extractor (특징 추출기) | Classifier (분류기) |
|---|---|---|
| 구성 | Conv + ReLU + Pool 블록 | Fully Connected Layer (FC) |
| 역할 | 이미지에서 유용한 특징 추출 | 추출된 특징으로 클래스 예측 |
| 출력 | 고차원 특징 벡터 | 클래스별 확률 분포 |
| 전이학습 | 다른 데이터셋에 재사용 가능 | 태스크별로 재학습 필요 |

이 구분은 **전이학습(Transfer Learning)**을 이해하는 데 핵심입니다. Feature Extractor는 엣지나 질감 같은 범용적인 시각 패턴을 학습하므로 다른 문제에도 재사용할 수 있지만, Classifier의 출력 크기는 클래스 수에 종속되기 때문에 새로운 문제마다 다시 학습시켜야 합니다.

### PyTorch 구현

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)     # 특징 추출
        x = self.classifier(x)   # 분류
        return x
```

`Dropout(0.5)`은 학습 시 뉴런의 50%를 무작위로 비활성화하여 과적합을 방지하는 정규화 기법입니다.

---

## 9. 하이퍼파라미터 설계 가이드 {#hyperparameter-guide}

### 세 가지 핵심 하이퍼파라미터

**필터 개수**: 적을수록 빠르지만 표현력이 감소하고, 많을수록 표현력이 증가하지만 과적합 위험이 커집니다. 일반적으로 32 → 64 → 128 → 256처럼 2배씩 증가시킵니다.

**커널 크기**:

- 3×3: VGG, ResNet에서 사용하는 가장 일반적인 크기
- 5×5, 7×7: AlexNet 등 초기 레이어에서 사용
- 1×1: 채널 수 조정, Bottleneck 구조에 활용

**레이어 깊이**: 얕은 네트워크는 간단한 특징만 학습하지만, 깊은 네트워크는 복잡한 특징을 학습할 수 있습니다. 단, 너무 깊으면 Gradient Vanishing 문제가 발생할 수 있습니다.

### 트레이드오프 분석

| 요소 | 작은 값 | 큰 값 |
|---|---|---|
| 커널 크기 | 파라미터 ↓, 층을 더 쌓을 여유 | 넓은 수용 영역, 파라미터 ↑ |
| 필터 개수 | 빠른 학습, 표현력 ↓ | 높은 표현력, 과적합 위험 |
| 깊이 | 간단한 특징 | 복잡한 특징, Vanishing 위험 |

### 정규화 기법

- **Dropout**: FC layer에 p=0.5 적용
- **Batch Normalization**: Conv layer 후에 적용
- **Data Augmentation**: 학습 데이터 증강

Batch Normalization은 각 미니배치의 활성화를 평균 0·분산 1로 정규화한 뒤 학습 가능한 스케일·시프트 파라미터를 다시 적용하는 기법입니다. 층마다 입력 분포가 흔들리는 것을 억제해 학습이 빠르고 안정적으로 진행되며, 약한 정규화 효과도 있습니다. 보통 `Conv → BN → ReLU` 순서로 배치합니다.

### 실전 가이드라인

- **Baseline**: 2-3 Conv blocks, 32→64→128 filters, 3×3 kernels
- **개선**: 레이어 추가, 필터 수 증가, 정규화 적용

---

## 10. 실전: CIFAR-10으로 CNN 학습시키기 {#cifar10-practice}

### CIFAR-10 데이터셋

- **60,000개**의 32×32 컬러 이미지 (훈련 50K / 테스트 10K)
- **10개 클래스**, 각 6,000장씩 균등 구성
- 저해상도라 빠른 실험이 가능하지만, 실제 사진이라 라벨 노이즈가 있고 cat vs dog, automobile vs truck처럼 클래스 간 유사성이 높아 생각보다 난이도가 있습니다.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) # 픽셀값 [0, 1] → [-1, 1]
```

픽셀값을 [-1, 1] 범위로 정규화하면 신경망 학습이 더 안정적이고 빠르게 수렴하는 경향이 있습니다. 여기서는 간단하게 0.5를 썼지만, 보통은 CIFAR-10의 채널별 실제 평균·표준편차(mean ≈ 0.4914, 0.4822, 0.4465 / std ≈ 0.247, 0.243, 0.261)를 쓰는 것이 더 일반적입니다.

### SimpleCNN 아키텍처

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32×32×3 → 32×32×32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 32×32×32 → 16×16×32

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 16×16×32 → 16×16×64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                             # 16×16×64 → 8×8×64
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 8×8×64 → 4096
            nn.Linear(64*8*8, 256),     # 4096 → 256
            nn.ReLU(),
            nn.Linear(256, 10)          # 256 → 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 파라미터 계산: FC 레이어가 대부분을 차지한다

| 레이어 | 계산식 | 파라미터 수 |
|---|---|---|
| Conv1 | 3×32×3×3 + 32 | 896 |
| Conv2 | 32×64×3×3 + 64 | 18,496 |
| FC1 | 4096×256 + 256 | 1,048,832 |
| FC2 | 256×10 + 10 | 2,570 |
| **Total** | | **~1.07M** |

![SimpleCNN 레이어별 파라미터 수. FC1 레이어가 약 104만 개로 전체의 약 98%를 차지하고, Conv1·Conv2는 각각 896개·18,496개에 불과하다](/assets/img/posts/cnn-convolution-complete-guide/05_parameter_distribution.png)

Conv 레이어는 필터가 이미지 전체에서 **공유**되기 때문에 파라미터 수가 매우 적습니다. 반면 FC 레이어는 입력과 출력의 모든 뉴런이 하나하나 연결되기 때문에, 4096×256처럼 큰 곱만큼 파라미터가 필요합니다. 실무에서 Global Average Pooling으로 큰 FC 레이어를 대체하는 이유가 바로 여기에 있습니다. `nn.Flatten()`과 `nn.Linear(64*8*8, 256)`을 `nn.AdaptiveAvgPool2d((1, 1))` + `nn.Flatten()` + `nn.Linear(64, 256)`으로 바꿔 보면, FC1 파라미터가 100만 개대에서 1만 6천 개대로 줄어드는 것을 직접 확인할 수 있습니다.

### 학습 루프 구현

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
```

핵심 흐름은 **예측(forward) → 손실 계산 → 기울기 계산(backward) → 가중치 업데이트(optimizer.step)**의 4단계이며, 이를 전체 데이터셋에 대해 한 바퀴 도는 것을 "1 에폭"이라 부릅니다.

### 주요 구성 요소

| 요소 | 선택 | 이유 |
|---|---|---|
| 손실 함수 | CrossEntropyLoss | 다중 클래스 분류에 적합. Softmax와 NLL Loss 결합 |
| 옵티마이저 | Adam (lr=0.001) | Adaptive learning rate로 빠른 수렴 |
| 배치 크기 | 64 | GPU 메모리와 학습 안정의 균형점 |
| 에폭 수 | 10-20 | CIFAR-10에서 충분한 학습을 위한 권장 범위 |

---

## 11. 모델 평가와 시각화 {#evaluation}

### 평가 함수 구현

```python
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total
```

**`model.eval()`**은 Dropout과 Batch Normalization을 평가 모드로 전환합니다. Dropout은 평가 시 모든 뉴런을 사용해야 하고, Batch Normalization은 학습 중 누적된 전체 데이터셋 이동 평균을 사용해야 하기 때문입니다.

**`torch.no_grad()`**는 Gradient 계산을 비활성화하여 메모리를 절약합니다. 평가 단계에서는 가중치를 업데이트하지 않으므로 기울기 추적이 불필요합니다.

### 학습 결과

앞의 SimpleCNN(Conv 2블록, BN·데이터 증강 없음)을 Adam(lr=0.001), 배치 64, 15 에폭으로 학습시키면 대략 다음과 같은 곡선을 그립니다.

| Epoch | Train Loss | Train Acc | Test Acc |
|---|---|---|---|
| 1 | 1.51 | 45.2% | 52.8% |
| 3 | 1.05 | 63.1% | 63.4% |
| 5 | 0.81 | 71.5% | 67.9% |
| 10 | 0.45 | 84.3% | 71.2% |
| 15 | 0.24 | 91.7% | 71.8% |

최종적으로 **테스트 정확도 약 72%** 수준입니다. 무작위 추측(10%)이나 완전결합 신경망 베이스라인보다는 확실히 높지만, 5 에폭 이후 Train Acc는 계속 오르는데 Test Acc는 71~72%에서 정체됩니다. Train과 Test 사이 약 20%p 차이는 전형적인 **과적합** 신호입니다.

여기서 성능을 끌어올리는 다음 수순이 바로 9장에서 정리한 정규화 기법입니다 — Conv 블록마다 Batch Normalization을 넣고, 학습 데이터에 `RandomCrop`·`RandomHorizontalFlip` 같은 증강을 적용하면 같은 구조로도 테스트 정확도를 80% 부근까지 올릴 수 있습니다.

### 시각화로 모델 이해하기

정확도 숫자 하나만 보고 끝내지 말고, 다음 네 가지 시각화로 모델을 다각도로 분석해보는 것이 좋습니다.

| 시각화 | 확인하는 것 |
|---|---|
| **학습 곡선** | 에폭에 따른 Loss/Accuracy 변화. 훈련·검증 곡선의 간격이 벌어지면 과적합 의심 |
| **예측 결과** | 실제 이미지와 정답·예측 레이블을 비교해 오분류 유형 분석 |
| **Confusion Matrix** | 어떤 클래스 쌍이 자주 혼동되는지 확인 (예: cat-dog, automobile-truck) |
| **Feature Map** | 각 합성곱 레이어의 활성화를 시각화. 초기 레이어는 엣지, 후기 레이어는 고수준 패턴 포착 |

이 네 가지를 종합하면, 단순히 "정확도 72%"라는 숫자를 넘어서 모델이 왜 그런 성능을 냈는지, 어디를 개선해야 하는지에 대한 실질적인 인사이트를 얻을 수 있습니다.

---

## 12. 마무리 {#summary}

이 글에서 다룬 CNN의 전체 흐름을 한 줄로 요약하면 다음과 같습니다.

> MLP는 이미지의 공간 구조를 무시해서 비효율적이었고, CNN은 **지역 연결성 · 파라미터 공유 · 계층적 학습**이라는 세 가지 아이디어로 이를 해결한다. 합성곱과 풀링을 반복해 특징을 추출(Feature Extractor)하고, FC 레이어로 최종 분류(Classifier)를 수행하는 것이 CNN의 기본 구조다.

**핵심 체크리스트**

- [ ] 합성곱은 필터를 슬라이딩하며 지역 패턴을 검출하는 연산이다
- [ ] Filter, Stride, Padding으로 출력 크기를 계산할 수 있다
- [ ] ReLU는 빠르고 효율적인 비선형성을 제공한다
- [ ] Pooling은 차원 축소, 위치 불변성, 수용 영역 확대를 동시에 제공한다
- [ ] 채널 수는 늘리고 공간 크기는 줄이는 것이 CNN 설계의 기본 패턴이다
- [ ] Feature Extractor는 재사용 가능하고, Classifier는 태스크별 재학습이 필요하다 (전이학습의 핵심)
- [ ] FC 레이어가 전체 파라미터의 대부분을 차지하는 경우가 흔하다

다음 단계로는 [ResNet, VGG 같은 사전 학습된 모델을 가져와 전이학습을 적용]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})해보거나, Batch Normalization과 Data Augmentation을 SimpleCNN에 직접 추가해 성능이 어떻게 개선되는지 실험해보는 것을 추천합니다.

---

*이 글은 CNN 등장 배경부터 PyTorch를 이용한 CIFAR-10 분류 실습까지의 학습 내용을 정리한 기술 블로그입니다.*
