---
title: "전이 학습으로 CIFAR-10 분류하기: ResNet-18 vs VGG-19-BN"
date: 2026-08-24 10:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, transfer-learning, resnet, vgg, cifar-10, machine-learning]
description: "사전 학습 모델을 활용한 전이 학습의 개념부터, ResNet-18과 VGG-19-BN의 레이어 교체 방식 차이, Adam과 SGD의 파인튜닝 성능 차이, CIFAR-10 실습 결과까지 정리했다."
---

> 이미지 분류 모델을 처음부터 학습시키려면 수백만 장의 데이터와 오랜 학습 시간이 필요합니다. 하지만 이미 세상의 다양한 사물을 학습한 모델이 있다면 어떨까요? 이 글에서는 **사전 학습 모델(Pretrained Model)**을 활용해 CIFAR-10 데이터셋을 분류하는 전이 학습(Transfer Learning) 과정을 정리합니다.

## 📌 목차

1. [사전 학습 모델이란](#pretrained-model)
2. [파인 튜닝 vs 전이 학습](#finetuning-vs-transfer)
3. [nn.AdaptiveAvgPool2d: 유연한 풀링 레이어](#adaptive-avgpool)
4. [데이터 준비](#data-preparation)
5. [ResNet-18 불러오고 레이어 교체하기](#resnet-setup)
6. [최적화 함수 선택: Adam vs SGD](#optimizer-choice)
7. [VGG-19-BN 활용하기](#vgg-setup)
8. [학습 결과 및 한계](#results)

---

## 1. 사전 학습 모델이란 {#pretrained-model}

[지난 글]({% post_url 2026-08-20-cnn-image-classification %})에서는 CNN이 이미지의 공간 정보를 유지하며 특징을 추출하는 원리를 다뤘습니다. 이번 글에서는 한 걸음 더 나아가, CNN을 처음부터 학습시키는 대신 이미 방대한 이미지로 훈련된 모델의 지식을 빌려오는 방법을 살펴봅니다.

사전 학습 모델은 ImageNet 같은 대규모 데이터셋으로 미리 훈련된 신경망입니다. 수백만 장의 이미지를 통해 이미 세상의 다양한 사물과 각도, 선·면·질감 같은 기본적인 시각적 특징을 학습한 상태죠.

마치 풍부한 지식을 갖춘 "언어의 달인"과 같습니다. 달인에게 새로운 전문 분야를 가르치면, 아무것도 모르는 사람보다 훨씬 빠르고 깊이 있게 배울 수 있습니다.

파이토치의 `torchvision.models`는 AlexNet, VGG, ResNet 등 유명한 모델들을 함수 호출 한 번으로 제공합니다.

![사전 학습 모델을 활용하면 이미 갖춘 지식으로 짧은 시간에 높은 초기 성능을 얻을 수 있고, 처음부터 학습하면 수백만 장의 데이터와 긴 학습 시간이 필요하다](/assets/img/posts/transfer-learning-resnet-vgg/01_pretrained_vs_scratch.svg)

### 사전 학습 모델 성능 비교

| 신경망 | 클래스명 | Top-1 에러 | Top-5 에러 |
|---|---|---|---|
| AlexNet | `alexnet` | 43.45 | 20.91 |
| VGG-19 with BN | `vgg19_bn` | 25.76 | 8.15 |
| ResNet-18 | `resnet18` | 30.24 | 10.92 |
| ResNet-152 | `resnet152` | 21.69 | 5.94 |

- **Top-1 에러**: 모델이 가장 높게 예측한 클래스가 정답이 아닌 확률
- **Top-5 에러**: 상위 5개 예측 클래스 안에 정답이 없을 확률

에러율은 낮을수록 좋은 성능을 의미하며, 일반적으로 네트워크가 깊고 복잡할수록 성능이 좋아지는 경향이 있습니다.

---

## 2. 파인 튜닝 vs 전이 학습 {#finetuning-vs-transfer}

사전 학습 모델을 내 문제에 적용하는 방식은 크게 두 가지로 나뉩니다.

| 구분 | 파인 튜닝 (Fine-Tuning) | 전이 학습 (Transfer Learning) |
|---|---|---|
| 방법 | 모든 파라미터를 우리 데이터로 재학습 | 대부분 동결, 출력에 가까운 파라미터만 학습 |
| 적합한 상황 | 데이터가 충분히 많을 때 | 데이터가 적을 때 |
| 특징 | 모델 전체를 새 작업에 맞게 조정 | 기존 지식을 최대한 보존 |
| 학습 시간 | 상대적으로 오래 걸림 | 상대적으로 짧음 |

![모든 레이어를 재학습하는 파인 튜닝과, 대부분의 레이어를 동결하고 출력층만 재학습하는 전이 학습의 구조 비교](/assets/img/posts/transfer-learning-resnet-vgg/02_finetuning_vs_transfer.svg)

핵심은 모델의 **"변화 범위"**에 있습니다. 파인 튜닝은 모델 전체를 새로운 작업에 맞게 최적화하려는 시도이고, 전이 학습은 사전 학습된 지식을 최대한 보존하며 최소한의 변경으로 새로운 작업에 적용하는 것을 목표로 합니다.

---

## 3. nn.AdaptiveAvgPool2d: 유연한 풀링 레이어 {#adaptive-avgpool}

`nn.AdaptiveAvgPool2d`는 CNN 모델의 '유연성'을 담당하는 중요한 레이어입니다.

- **기존 MaxPool2d**: 커널 크기를 지정하여 입력 이미지 크기에 따라 출력 크기가 결정됨
- **AdaptiveAvgPool2d**: 원하는 출력 크기를 직접 지정하여 입력 크기와 무관하게 일정한 출력을 생성

예를 들어 입력이 `(100, 32, 16, 16)`이든 `(100, 32, 8, 8)`이든, `AdaptiveAvgPool2d(1,1)`을 거치면 항상 `(100, 32, 1, 1)`로 변환됩니다.

![입력 해상도가 16x16이든 8x8이든 AdaptiveAvgPool2d를 거치면 항상 동일한 (1,1) 크기로 출력되는 과정](/assets/img/posts/transfer-learning-resnet-vgg/03_adaptive_avgpool.svg)

이 레이어 덕분에 내 데이터셋의 이미지 크기가 사전 학습 모델의 원본 입력 크기와 다르더라도, 모델 뒷단(분류기)은 항상 고정된 크기의 입력을 받게 되어 크기 걱정 없이 모델을 재활용할 수 있습니다.

---

## 4. 데이터 준비 {#data-preparation}

CIFAR-10 데이터는 원래 32x32 픽셀이지만, 사전 학습 모델은 대부분 224x224 같은 더 큰 이미지로 학습되었습니다. 학습 시간과 성능의 균형을 고려해 이미지 크기를 **112x112**로 조정합니다.

**학습 데이터 변환**
- 리사이즈 (112x112)
- 좌우 반전 (p=0.5) → 데이터 증강
- 정규화
- 랜덤 지우기 (p=0.5) → 과적합 방지

**검증 데이터 변환**
- 리사이즈 (112x112)
- 정규화

검증 데이터에는 증강(반전, 지우기)을 적용하지 않습니다. 모델의 실제 성능을 있는 그대로 평가해야 하기 때문입니다. 배치 사이즈는 50으로 설정하여 메모리 효율성과 학습 안정성을 확보합니다.

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5),
])

val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 5. ResNet-18 불러오고 레이어 교체하기 {#resnet-setup}

ResNet-18은 VGG-19-BN과 달리 마지막 fc(fully connected) 레이어에 바로 접근할 수 있어 레이어 교체 방식이 더 간단합니다.

### 왜 ResNet-18은 안정적으로 깊게 쌓일 수 있을까

일반 신경망은 층이 깊어질수록 입력 신호가 여러 층을 거치며 점점 약해져 **기울기 소실** 문제가 생깁니다. ResNet은 입력을 다음 층 출력에 그대로 더해주는 **스킵 커넥션(잔차 연결)** 덕분에 층이 아무리 깊어져도(152층까지) 기울기가 안정적으로 전달됩니다.

![층이 깊어질수록 신호가 약해지는 일반 신경망과, 스킵 커넥션으로 입력을 그대로 더해 기울기를 안정적으로 전달하는 ResNet의 구조 비교](/assets/img/posts/transfer-learning-resnet-vgg/04_resnet_residual.svg)

### 코드로 구현하기

**1단계. 모델 불러오기**
```python
net = models.resnet18(pretrained=True)
```

**2단계. 최종 레이어 확인**
```python
in_features = net.fc.in_features  # 512
```

**3단계. 레이어 교체**
```python
net.fc = nn.Linear(in_features, n_output)  # n_output = 10 (CIFAR-10 클래스 수)
```

---

## 6. 최적화 함수 선택: Adam vs SGD {#optimizer-choice}

모델을 학습시킬 때 가중치를 어떻게 업데이트할지 결정하는 최적화 함수(Optimizer) 비교입니다.

| 구분 | Adam | SGD with Momentum |
|---|---|---|
| 학습률 | 파라미터별 자동 조정 | 고정 |
| 초반 속도 | 빠름 | 상대적으로 느림 |
| 파인 튜닝 안정성 | 불안정할 수 있음 | 더 안정적 |
| 일반화 성능 | 상대적으로 낮을 수 있음 | 더 나은 경향 |

특히 **파인 튜닝처럼 이미 잘 학습된 가중치를 미세 조정하는 상황**에서는 SGD with Momentum이 Adam보다 더 나은 결과를 보이는 경향이 있습니다.

**이유**

1. **적응적 학습률의 부작용**: Adam은 기울기가 작다고 판단되면 학습률을 오히려 크게 키우는 경우가 있어, 이미 잘 잡혀 있는 가중치를 필요 이상으로 흔들 위험이 있습니다.
2. **Sharp vs Flat Minima**: Adam은 좁고 뾰족한 최솟값(sharp minima)에, SGD는 넓고 완만한 최솟값(flat minima)에 수렴하는 경향이 있습니다. Flat minima는 일반화에 더 유리합니다.
3. **모멘텀의 관성 효과**: 이전 기울기 방향을 누적해 노이즈에 덜 흔들리고 일관된 방향으로 수렴합니다.
4. **정규화 처리 방식**: SGD는 weight decay가 기울기에 직접적이고 일관되게 반영되어 과적합 방지 효과가 더 예측 가능합니다.

> Adam은 "빠르게 배우는 데" 최적화되어 있고, SGD with Momentum은 "안정적으로 좋은 지점에 정착하는 데" 최적화되어 있습니다. 파인 튜닝은 이미 좋은 위치에서 시작해 조금씩 다듬는 작업이라 SGD의 안정적인 수렴 특성이 유리하게 작용합니다.

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
```

---

## 7. VGG-19-BN 활용하기 {#vgg-setup}

VGG-19-BN은 ResNet-18과 다른 구조를 가지고 있어 레이어 교체 방식이 조금 다릅니다.

- VGG-19-BN의 분류기는 `classifier`라는 **Sequential 모듈**로 구성되어 있습니다.
- 마지막 선형 레이어는 `classifier`의 **6번째 요소(인덱스 6)**에 위치합니다.
- 입력 차원은 4096, 출력 차원은 ImageNet 기준 1000입니다.

![ResNet-18은 net.fc 속성으로 바로 접근해 교체하고, VGG-19-BN은 classifier Sequential의 6번째 요소를 인덱스로 찾아 교체하는 구조 비교](/assets/img/posts/transfer-learning-resnet-vgg/05_resnet_vs_vgg_structure.svg)

```python
net = models.vgg19_bn(pretrained=True)

in_features = net.classifier[6].in_features  # 4096
net.classifier[6] = nn.Linear(in_features, n_output)
```

### ResNet-18 vs VGG-19-BN 트레이드오프 종합 비교

| 항목 | ResNet-18 | VGG-19-BN |
|---|---|---|
| 아키텍처 특징 | 잔차 연결 | 순차 스택 + Batch Normalization |
| 네트워크 깊이 | 18층 | 19층 |
| 레이어 접근 방식 | `net.fc` (속성) | `net.classifier[6]` (인덱스) |
| fc 입력 차원 | 512 | 4096 |
| 파라미터 수 | 적음 | 많음 |
| 에폭당 학습 시간 | 짧음 | 약 8분 (더 오래 걸림) |
| 기대 정확도 | 상대적으로 낮을 수 있음 | 더 높은 정확도 기대 가능 |

두 모델 모두 동일한 하이퍼파라미터로 학습해 공정하게 비교합니다.

- 학습률: 0.001
- 최적화 함수: SGD with Momentum (0.9)
- 손실 함수: CrossEntropyLoss
- 에폭 수: 5

---

## 8. 학습 결과 및 한계 {#results}

VGG-19-BN을 5 에폭 파인 튜닝한 결과입니다.

| 지표 | 값 |
|---|---|
| 초기 정확도 | 93.67% |
| 최종 정확도 | 95.76% |
| 성능 향상 | +2.09%p |

![VGG-19-BN을 5 에폭 파인 튜닝했을 때 정확도가 93.67%에서 95.76%로 향상되는 막대그래프](/assets/img/posts/transfer-learning-resnet-vgg/06_performance_summary.svg)

학습 곡선을 보면 훈련·검증 손실이 에폭이 진행될수록 꾸준히 감소하고, 검증 손실이 훈련 손실보다 낮게 유지되어 과적합 없이 안정적으로 학습되었음을 확인할 수 있습니다. 정확도 역시 훈련·검증 모두 꾸준히 상승하며 5에폭 시점에는 95~96% 부근에서 수렴했습니다.

파인 튜닝 이전(93.67%)에도 이미 90%를 넘는 정확도를 보였다는 점이 인상적입니다. ImageNet에서 학습된 특징 추출기가 CIFAR-10에도 상당 부분 그대로 통했다는 뜻이고, 5 에폭만 미세 조정해도 +2.09%p의 추가 향상을 얻을 수 있었습니다.

### 한계

이 실습 결과를 해석할 때 유의할 점도 있습니다.

- **ResNet-18과의 직접 비교 부재**: 이번 실습에서는 VGG-19-BN만 파인 튜닝해 결과를 측정했습니다. 7절의 "기대 정확도" 비교는 일반적인 아키텍처 트레이드오프에 근거한 예상치일 뿐, 동일 조건에서 ResNet-18을 직접 학습시켜 얻은 실측값은 아닙니다. 두 모델을 공정하게 비교하려면 ResNet-18도 동일한 하이퍼파라미터로 학습시켜 최종 정확도를 함께 측정해야 합니다.
- **짧은 학습 기간**: 5 에폭만 학습했기 때문에 검증 정확도가 아직 완전히 수렴했다고 단정하기는 이릅니다.
- **단일 실행 결과**: seed를 고정하지 않고 한 번만 학습한 결과라면, 실행마다 초기화·데이터 순서에 따라 수치가 소폭 달라질 수 있습니다.
- **다운스케일된 입력**: 224x224가 아닌 112x112로 리사이즈했기 때문에, 사전 학습 당시의 원본 해상도보다 정보 손실이 있는 상태로 학습된 결과입니다.

---

## 핵심 정리

- **사전 학습 모델**: 대규모 데이터셋으로 미리 훈련된 모델을 활용해 학습 시간을 단축하고 성능을 향상시킬 수 있다.
- **파인 튜닝**: 사전 학습된 가중치를 새로운 작업에 맞게 미세 조정하며, SGD with Momentum이 효과적이다.
- **모델 선택**: ResNet-18은 파라미터가 적어 학습이 빠르고, VGG-19-BN은 구조가 복잡한 만큼 학습은 오래 걸리지만 이번 실습에서 90%대 후반의 정확도를 보였다. 데이터·시간 예산에 맞게 선택하자.

---

## Appendix

### 파인 튜닝 vs 전이 학습 (개념 재정리)

파인 튜닝은 사전 학습 모델의 모든 파라미터를 새로운 데이터에 맞게 미세 조정하여 최적화하는 방식입니다. 반면 전이 학습은 모델의 가중치 대부분을 고정하고 출력 레이어와 같은 극히 일부분만 학습시키는, 데이터가 적을 때 더 보수적이고 효율적인 접근법입니다.

### 재현성 확보: torch seed

머신러닝 모델의 학습 과정에는 난수가 광범위하게 사용됩니다. 초기 가중치 설정, 데이터 샘플링, 드롭아웃 등 여러 단계에서 예측 불가능한 변화를 가져올 수 있습니다.

seed를 고정하는 것은 이러한 난수 생성 과정을 미리 정해진 순서에 따라 일관되게 만드는 것을 의미합니다. 파이토치에서 완벽한 재현성을 보장하려면 CPU와 GPU의 난수 생성기에 동일한 시드를 설정하고, GPU의 비결정적 알고리즘 사용을 비활성화하는 설정을 추가해야 합니다.

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### 풀링 레이어를 제거하는 이유

VGG-19-BN과 같은 일부 사전 학습 모델을 CIFAR-10과 같은 특정 데이터셋에서 파인 튜닝할 때, 단순히 마지막 분류 레이어만 교체하는 것을 넘어 풀링 레이어를 제거하는 경우가 발생할 수 있습니다.

이는 모델의 내부 구조를 특정 환경이나 요구사항에 맞게 최적화하기 위함입니다. 원본 모델이 가진 특정 풀링 레이어가 새로운 작업의 입력 크기나 특징 추출 방식에 비효율적이거나 오히려 성능을 저해할 수 있기 때문입니다. 모델 배포 시 리소스 제약이나 특정 하드웨어 환경에 맞춰 모델을 경량화하거나 불필요한 연산을 줄이는 과정에서 이런 고급 기술이 필요해집니다.

---

*이 글은 전이 학습(Transfer Learning) 기반 CIFAR-10 이미지 분류 실습 자료를 정리한 것입니다. 코드 예제는 실제 동작을 위해 데이터 경로, 모델 하이퍼파라미터 등을 환경에 맞게 조정해야 할 수 있습니다.*
