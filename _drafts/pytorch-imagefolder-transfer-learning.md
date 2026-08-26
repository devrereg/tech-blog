---
title: "사용자 데이터로 이미지 분류 모델 만들기: ImageFolder부터 전이 학습까지"
categories: [AI, Deep Learning]
tags: [pytorch, transfer-learning, fine-tuning, image-classification, cnn]
description: "PyTorch ImageFolder로 사용자 데이터셋을 구성하고, 파인튜닝과 전이 학습의 차이를 개미/벌·허스키/늑대 실습으로 비교 정리한다."
---

실무에서 이미지 분류 모델을 만들 때는 MNIST나 CIFAR처럼 이미 정리된 데이터셋이 아니라, 우리가 직접 수집한 JPEG/PNG 이미지를 다뤄야 합니다. 이 글에서는 PyTorch의 `ImageFolder`로 사용자 데이터셋을 구성하는 방법부터, **파인튜닝(Fine-tuning)**과 **전이 학습(Transfer Learning)**의 차이, 그리고 극소량 데이터에서도 높은 성능을 낸 실제 사례까지 정리합니다.

## 1. 오늘 다룰 내용

- 문제 정의하기
- 데이터 준비
- 파인튜닝의 경우
- 전이 학습의 경우
- 사용자 정의 데이터를 사용하는 경우

즉, **분류 문제를 정의하고 → 데이터를 준비하고 → 기존 모델을 활용(파인튜닝/전이학습)하거나 직접 만든 데이터로 학습시키는 방법**을 다룹니다.

## 2. 사전학습모델과 ImageFolder

PyTorch의 `ImageFolder` 기능을 활용하면 **데이터셋을 구성하는 부분만 변경**하고 나머지 학습 코드는 그대로 재사용할 수 있습니다.

`ImageFolder`는 특정 트리 구조로 배치된 이미지 파일을 머신러닝의 입력 데이터로 자동 변환해줍니다. **폴더 이름을 라벨로 인식**하고, 해당 폴더 아래의 이미지들을 그 라벨의 데이터로 사용해 PyTorch 데이터셋을 생성합니다.

### ImageFolder 필수 디렉터리 구조

`ImageFolder`를 사용하려면 학습 데이터를 아래와 같은 트리 구조로 배치해야 합니다. `train`(훈련) 폴더와 `val`(검증) 폴더를 만들고, 각 폴더 아래에 분류하려는 클래스 이름으로 하위 폴더를 생성합니다.

![ImageFolder 디렉터리 구조](/assets/img/posts/pytorch-imagefolder-transfer-learning/folder_structure.png)

```text
root/
 ├── train/
 │    ├── ants/
 │    │    ├── 0013035.jpg
 │    │    └── ...
 │    └── bees/
 │         ├── 1092977343_cb42b38d62.jpg
 │         └── ...
 └── val/
      ├── ants/
      │    ├── 10308379_1b6c72e180.jpg
      │    └── ...
      └── bees/
           ├── 1032546534_06907fe3b3.jpg
           └── ...
```

**핵심 규칙은 단 하나입니다.** 폴더 이름 = 라벨, 폴더 속 이미지 = 데이터. 이 구조만 지키면 `ImageFolder`가 알아서 학습용 데이터셋으로 변환해줍니다.

## 3. 데이터 전처리: 훈련용 vs 검증용 Transform

훈련 데이터와 검증 데이터는 목적이 다르기 때문에 전처리 방식도 다르게 적용해야 합니다.

**검증 데이터 전처리 (일관성 중심)**
- `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize`
- 사전 학습된 모델이 224×224 크기로 학습되었기 때문에, 일관된 평가를 위해 항상 이미지의 **중심부**를 사용합니다.

**훈련 데이터 전처리 (다양성 중심, 데이터 증강)**
- `RandomResizedCrop(224)` → 무작위 크기로 잘라내고 리사이즈
- `RandomHorizontalFlip` → 무작위 좌우 반전
- `RandomErasing` → 무작위 영역 지우기

이런 데이터 증강 기법을 사용하면 모델이 매번 조금씩 다른 이미지를 학습하게 되어 **과적합을 방지**하고 일반화 성능을 높일 수 있습니다.

> **정리:** 검증 = 항상 같은 방식(CenterCrop)으로 공정하게 평가하고, 훈련 = 매번 다른 방식(Random~)으로 다양한 경험을 쌓아 더 튼튼한 모델을 만듭니다.

## 4. ImageFolder로 데이터셋 생성하기

디렉터리 경로를 지정하고 `ImageFolder`를 호출하면 바로 데이터셋이 만들어집니다. 개미(ants)와 벌(bees)을 분류하는 예제는 **훈련 데이터 244건, 검증 데이터 153건, 총 2개 클래스**로 구성됩니다.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

print(len(train_dataset), len(val_dataset), train_dataset.classes)
# 244 153 ['ants', 'bees']
```

이렇게 적은 데이터셋(397장)으로도 학습이 가능한 이유는 **사전 학습 모델**을 활용하기 때문입니다. 처음부터 학습한다면 이 정도 데이터로는 부족합니다.

`DataLoader`는 배치 크기 10으로 설정해 학습 데이터를 효율적으로 공급하고, `shuffle=True`로 설정해 매 epoch마다 데이터 순서를 섞어 모델이 데이터 순서를 학습하는 것을 방지합니다.

## 5. 파인튜닝 vs 전이 학습

두 가지 방식 모두 사전 학습 모델을 재사용하지만, **어느 파라미터를 업데이트하는지**에서 근본적인 차이가 있습니다. 두 개념에 대한 더 자세한 설명과 ResNet-18/VGG-19-BN 비교는 지난 글 [전이 학습으로 CIFAR-10 분류하기: ResNet-18 vs VGG-19-BN]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})에서 다뤘으므로, 여기서는 핵심만 짚고 개미/벌 데이터로 직접 실습한 코드와 결과에 집중합니다.

> 이 글에서는 편의상 "파인튜닝 = 모든 파라미터 재학습", "전이 학습 = 대부분 동결 후 마지막 계층만 학습"으로 나눠 부릅니다. 엄밀히는 전이 학습이 상위 개념이고 파인튜닝은 그 안의 한 전략이지만, 두 접근을 대조하는 것이 이 글의 목적이라 편의상 병렬로 표기합니다.

![파인튜닝 vs 전이학습 구조 비교](/assets/img/posts/pytorch-imagefolder-transfer-learning/finetuning_vs_transfer.png)

### 파인튜닝 (Fine-tuning)

사전 학습된 모델의 **모든 파라미터**를 우리 데이터에 맞게 다시 조정하는 방식입니다. VGG-19-BN 모델을 사용한 개미/벌 분류 실습 과정은 다음과 같습니다.

```python
from torchvision import models
import torch.nn as nn

model = models.vgg19_bn(pretrained=True)

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 개미 vs 벌, 2개 클래스

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

1. **사전학습모델 불러오기** → VGG-19-BN의 학습이 끝난 파라미터와 함께 불러옵니다.
2. **최종 계층 수정** → 분류 클래스가 2개이므로 `classifier[6]`의 출력을 2로 변경합니다.
3. **최적화 함수 설정** → `model.parameters()`, 즉 모델의 모든 파라미터를 업데이트 대상으로 지정합니다.
4. **학습 진행** → 5 에포크 동안 학습을 진행해 모델을 미세 조정합니다.

**결과:** 5 에포크 학습 후 검증 정확도 **96.08%**, 훈련 정확도 **94.26%**를 달성했습니다. 검증 정확도가 훈련 정확도보다 높아, 과적합 징후 없이 안정적인 성능을 보였습니다.

### 전이 학습 (Transfer Learning)

사전 학습 모델의 **대부분 가중치는 그대로 동결**하고, **마지막 분류 계층만 새로 학습**하는 방식입니다. 학습 데이터가 매우 적을 때 효과적입니다.

```python
model = models.vgg19_bn(pretrained=True)

for param in model.parameters():
    param.requires_grad = False  # 모든 가중치 동결

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 새로 교체한 계층은 기본값 requires_grad=True

optimizer = torch.optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)
```

1. **가중치 동결** → 모든 파라미터의 `requires_grad`를 `False`로 설정해 기존 학습된 가중치가 업데이트되지 않도록 동결합니다.
2. **최종 계층 교체** → 마지막 분류 계층만 새로운 `Linear` 계층으로 교체해 2개 클래스 분류에 맞게 조정합니다. 새로 만든 계층은 기본적으로 `requires_grad=True` 상태라 학습 대상에 그대로 포함됩니다.
3. **선택적 학습** → 옵티마이저에 최종 계층의 파라미터(`model.classifier[6].parameters()`)만 전달해 해당 계층만 학습되도록 합니다.

**결과:** 전이 학습 결과 역시 약 **96%**의 정확도를 보여, 이 예제에서는 파인튜닝과 동일한 성능을 기록했습니다.

### 두 방식 비교

| 구분 | 파인튜닝 | 전이 학습 |
|---|---|---|
| 업데이트 대상 | 모든 파라미터 | 최종 계층만 |
| 필요 데이터셋 | 더 많이 필요 | 적어도 가능 |
| 학습 시간 | 더 오래 걸림 | 더 짧음 |
| 효과적인 상황 | 데이터가 충분할 때 | 데이터가 적을 때 |

일반적으로 **학습 데이터가 적을수록 전이 학습이 더 좋은 성능**을 내는 것으로 알려져 있습니다. 데이터가 적으면 전이 학습을, 데이터가 충분하다면 파인튜닝을 우선 고려하는 것이 실무 판단 기준입니다.

## 6. 실제 사례: 극소량 데이터로 어려운 문제 풀기 (허스키 vs 늑대)

앞선 개미/벌 예제보다 훨씬 까다로운 문제로 배운 내용을 검증해봅니다. **시베리안 허스키와 늑대**는 외형이 매우 유사해 분류가 까다롭고, 데이터 수도 훨씬 적습니다.

**전략 3가지**

- **극소량 데이터**: 훈련 데이터 40장, 검증 데이터 10장 (개미/벌 예제의 397장보다 훨씬 적음)
- **전이 학습 적용**: 데이터 수가 매우 적으므로, 5절과 동일한 VGG-19-BN에 5절과 같은 전이 학습(가중치 동결 + 마지막 계층만 교체) 코드를 그대로 적용합니다.
- **데이터 증강 강화**: `RandomHorizontalFlip`과 `RandomErasing`을 추가해 데이터 부족 문제를 완화합니다.

배경이 상당 부분 제거된 이미지를 사용하므로 224×224 크기로 바로 리사이즈하며, 배치 사이즈는 데이터 수가 적어 5로 설정합니다.

### 학습 결과

![허스키 vs 늑대 에포크별 정확도 추이 (전이학습)](/assets/img/posts/pytorch-imagefolder-transfer-learning/epoch_curve.png)

| Epoch | 훈련 정확도 | 검증 정확도 |
|---|---|---|
| 1 | 65% | 100% |
| 2 | 85% | 90% |
| 9 | 100% | 100% |
| 10 (최종) | 92.5% | **100%** |

검증 데이터가 10장으로 매우 적기 때문에 초반부터 검증 정확도가 오르내리며 지표가 다소 불안정하게 보입니다. 훈련 정확도도 9번째 에포크에서 100%까지 올랐다가 마지막 에포크에서 92.5%로 다시 떨어지는데, 훈련 데이터가 40장(배치 크기 5로 총 8배치)뿐인 극소형 데이터셋이라 배치 구성에 따라 수치가 출렁일 수 있기 때문입니다. 다만 검증 정확도는 흔들림 없이 **검증 데이터 10장 모두를 정확하게 분류**하는 완벽한 성능으로 마무리됐습니다. 전이 학습이 극소량 데이터에서도 효과적임을 입증한 사례입니다.

## 7. 세 가지 실습 결과 한눈에 보기

지금까지 진행한 개미/벌(파인튜닝, 전이학습), 허스키/늑대(전이학습) 세 가지 실습의 정확도를 비교하면 다음과 같습니다.

![실습별 학습 결과 비교](/assets/img/posts/pytorch-imagefolder-transfer-learning/accuracy_comparison.png)

데이터가 충분한 개미/벌 예제에서는 파인튜닝과 전이학습이 비슷한 성능을 냈지만, **데이터가 극히 적었던 허스키/늑대 예제에서는 전이학습이 오히려 100%라는 더 높은 검증 정확도**를 기록했습니다. (다만 검증 데이터가 10장뿐이라는 점에서, 이 수치는 "완벽한 일반화 성능"이라기보다 "소규모 데이터에서도 전이 학습이 잘 작동한다는 경향성"으로 이해하는 것이 좋습니다.)

## 8. 핵심 정리

**① ImageFolder 활용**
특정 디렉터리 구조(`train`/`val` → 클래스명 폴더 → 이미지)만 지키면 자동으로 데이터셋을 생성해 실무에서 빠르게 적용할 수 있습니다.

**② 데이터 증강의 중요성**
`RandomErasing`, `RandomHorizontalFlip` 등의 기법으로 적은 데이터로도 높은 성능을 달성할 수 있습니다.

**③ 전이 학습**
데이터가 적은 경우 전이 학습이 파인튜닝보다 효과적이며, 40장의 극소량 데이터로도 100% 정확도를 달성한 사례를 확인했습니다.

> 사전 학습된 모델과 적절한 학습 전략을 선택하면 실무에서 마주치는 다양한 이미지 분류 문제를 효과적으로 해결할 수 있습니다. **데이터의 양과 특성에 따라 파인튜닝과 전이 학습 중 적절한 방법을 선택하는 것**이 중요합니다.

### 실무 의사결정 가이드

앞의 두 사례는 모두 "데이터 양"만을 기준으로 판단했지만, 실무에서는 사전 학습 모델이 원래 학습한 도메인과 우리 문제가 얼마나 비슷한지도 함께 고려하는 것이 일반적입니다.

- 데이터가 **충분하고** 문제가 사전 학습 모델의 원래 학습 도메인과 **많이 다르다** → 파인튜닝 고려
- 데이터가 **적고** 문제가 기존 모델이 이미 익힌 특징과 **유사하다** → 전이학습 우선 고려

---

*본 포스트는 "사용자 데이터 분류" 수업 자료(개미/벌, 허스키/늑대 이미지 분류 실습)를 바탕으로 정리한 기술 블로그입니다.*
