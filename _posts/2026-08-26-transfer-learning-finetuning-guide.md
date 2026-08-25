---
title: "전이 학습 파인튜닝 완전 정리: 동결 범위·학습률·데이터 증강 전략"
date: 2026-08-26 10:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, transfer-learning, fine-tuning, resnet, data-augmentation, machine-learning]
description: "전이 학습 파인튜닝에서 층별 동결 범위, 차등 학습률, Gradual Unfreezing, 데이터 증강 강도를 어떻게 정할지 예시 수치와 함께 정리하고, 데이터 크기·도메인 유사도별 전략 선택 가이드까지 다뤘다."
---

> [지난 글]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})에서는 ResNet-18과 VGG-19-BN으로 전이 학습을 실습하며 "레이어를 어떻게 교체하는가"를 다뤘습니다. 이번 글은 한 단계 더 들어가, 사전 학습 모델을 내 데이터에 맞게 미세조정(fine-tuning)할 때 마주치는 실전 질문들 — 어디까지 학습시키고, 학습률은 층마다 어떻게 다르게 줄지, 데이터 증강은 얼마나 세게 걸지 — 을 예시 수치와 함께 정리합니다.
>
> ⚠️ 아래 나오는 정확도·과적합 수치는 실제로 학습을 돌려 얻은 벤치마크가 아니라, 각 전략이 어떤 방향으로 차이를 만드는지 직관적으로 보여주기 위한 예시 수치입니다. 실제 프로젝트에 적용할 때는 반드시 자신의 데이터로 직접 검증해야 합니다.

사전 학습 모델을 내 데이터에 맞게 파인튜닝할 때, 우리는 늘 같은 질문과 마주칩니다.

- 모델의 어느 부분을 학습시키고, 어느 부분을 얼려야 할까?
- 학습률(Learning Rate)은 층마다 똑같이 줘야 할까, 다르게 줘야 할까?
- 데이터 증강은 얼마나 세게 적용해야 할까?
- 데이터가 아주 적으면 어떻게 해야 할까?

이 글은 ResNet-50 + ImageNet 전이 학습을 기준으로, 이 질문들에 대한 답을 정리합니다. "이론 → 코드 → 예시 수치"의 순서로, 적은 데이터에서도 안정적인 성능을 내기 위한 하나의 일관된 워크플로를 만드는 것이 목표입니다.

## 📌 목차

1. [층별 동결 전략: 얼마나 많이 풀어야 하는가](#freeze-strategy)
2. [차등 학습률: 층마다 다른 속도로 학습시키기](#differential-lr)
3. [Gradual Unfreezing: 시간에 따라 점진적으로 풀어주기](#gradual-unfreezing)
4. [데이터 증강 강도: 얼마나 세게 흔들 것인가](#data-augmentation)
5. [작은 데이터셋을 위한 종합 프로토콜](#small-dataset-protocol)
6. [종합: 상황별 전략 선택 가이드](#decision-guide)

---

## 1. 층별 동결 전략: 얼마나 많이 풀어야 하는가 {#freeze-strategy}

전이 학습에서 가장 먼저 결정해야 할 것은 **"백본(backbone)의 어느 부분을 학습 가능하게 열어둘 것인가"**입니다. 이 결정에 따라 크게 3가지 전략으로 나뉩니다.

- **완전 동결 (Full Freeze)**: 백본 전체를 고정하고, 새로 붙인 분류기(classifier)만 학습
- **부분 해제 (Partial Fine-tuning)**: 이전 층은 고정하고, 후반부 몇 개 블록만 학습
- **전체 해제 (Full Fine-tuning)**: 모든 층을 학습 가능하게 설정

![층별 동결 전략 3가지를 비교한 다이어그램. Full Freeze는 fc만 학습, Partial Fine-tuning은 layer3·layer4·fc를 학습, Full Fine-tuning은 모든 층을 학습한다](/assets/img/posts/transfer-learning-finetuning-guide/01_freeze_strategies.svg)

| 전략 | 장점 | 단점 |
|---|---|---|
| 완전 동결 | 빠른 학습, 과적합 방지 효과 | 도메인 차이가 클 때 성능 제한 |
| 부분 해제 | 균형 잡힌 접근, 안정적 | k(해제 블록 수) 선택이 중요 |
| 전체 해제 | 최대 표현력과 유연성 | 과적합 위험, 느린 학습 |

### 왜 후반부 층만 다시 학습시키는 게 효과적일까

CNN의 층은 "일반적인 특징 → 구체적인 특징" 순서로 학습됩니다.

- **초반 층**: 선, 모서리, 색깔, 질감 같은 범용적인 시각 패턴 → 어떤 이미지에도 적용
- **후반 층**: 초반 패턴을 조합한 구체적·추상적 개념 → **원래 학습 작업(ImageNet 분류)에 특화**

내 데이터의 클래스가 ImageNet과 완전히 겹치지 않는다면, 후반 층은 "재조정"이 필요합니다. 반면 초반 층은 이미 충분히 범용적이라, 굳이 건드리면 오히려 과적합만 키우고 가지고 있던 좋은 특징을 망각시키는(catastrophic forgetting) 위험이 커집니다. 그래서 **"딱 필요한 부분(후반 층)만 선택적으로 골라서 여는 것"**이 핵심입니다.

### PyTorch 구현

```python
import torch
import torch.nn as nn
from torchvision import models

# 전략 1: 완전 동결
def setup_full_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 분류기만 학습 가능
    return model

# 전략 2: 부분 해제 (last k blocks)
def setup_partial_finetune(model, k=1):
    for param in model.parameters():
        param.requires_grad = False
    if k >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if k >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 전략 3: 전체 해제
def setup_full_finetune(model):
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

세 함수 모두 원리는 동일합니다. `requires_grad` 속성 하나로 "이 파라미터를 학습시킬지 말지"를 결정합니다. 이 단순한 스위치가 모델의 학습 방식을 근본적으로 바꿉니다.

### 예시로 보는 전략별 성능 차이

**가정한 조건 (예시 수치)**: ResNet-50 (ImageNet pre-trained), 커스텀 데이터셋(5 classes, 500 images/class = 총 2,500장), Train/Val 80/20, 30 epochs, SGD momentum 0.9

![동결 전략별 예시 결과 막대그래프. Val Accuracy는 Last 2 Blocks에서 89.1%로 최고를 기록했고, Overfit Gap은 Full Fine-tune에서 12.4%로 가장 컸다](/assets/img/posts/transfer-learning-finetuning-guide/02_freeze_experiment.svg)

| Strategy | Val Accuracy | Train Time | Overfit Gap |
|---|---|---|---|
| Full Freeze | 82.5% | 1.2 min | 3.2% |
| Last 1 Block | 87.3% | 2.1 min | 5.8% |
| **Last 2 Blocks** | **89.1%** | 3.5 min | 8.1% |
| Full Fine-tune | 88.6% | 5.2 min | **12.4%** |

**이 예시 수치가 보여주는 패턴**: 무조건 많이 풀어서 학습시킨다고 성능이 좋아지는 게 아니라는 것입니다. 위 예시에서는 Full Fine-tune(88.6%)이 오히려 Last 2 Blocks(89.1%)보다 낮고, 과적합 정도(Overfit Gap)는 12.4%로 가장 큽니다. 데이터가 2,500장으로 적은 상황에서는 **"필요한 만큼만 여는 것"**이 전체를 다 여는 것보다 나은 결과로 이어지는 경우가 많습니다.

---

## 2. 차등 학습률: 층마다 다른 속도로 학습시키기 {#differential-lr}

동결 여부가 "학습할지 말지"의 이진 선택이었다면, **차등 학습률(Differential Learning Rate)**은 "얼마나 강하게 학습시킬지"를 연속적으로 조정하는 더 정교한 기법입니다.

### 왜 필요한가

사전 학습된 백본은 이미 좋은 특징을 가지고 있어 큰 변화가 불필요하지만, 새로 만든 분류기는 랜덤 초기화 상태라 큰 변화가 필요합니다. 그런데 모든 층에 동일한 학습률을 쓰면 둘 다 망가지기 쉽습니다.

- **LR을 크게 설정** → 분류기 학습은 좋지만 백본이 파괴됨 (catastrophic forgetting)
- **LR을 작게 설정** → 백본은 안전하지만 분류기 학습이 너무 느림

해결책은 **층의 위치에 따라 학습률을 다르게 주는 것**입니다.

![차등 학습률 개념도. Early Backbone은 lr×0.1, Mid Backbone은 lr×1, Late Backbone은 lr×3, Classifier Head는 lr×10으로 출력에 가까울수록 학습률이 커진다](/assets/img/posts/transfer-learning-finetuning-guide/03_differential_lr.svg)

### PyTorch 구현: Parameter Groups

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 층별로 다른 학습률 지정
params = [
    {'params': model.layer1.parameters(), 'lr': 1e-4},  # Early layers
    {'params': model.layer2.parameters(), 'lr': 3e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 3e-3},  # Late layers
    {'params': model.fc.parameters(), 'lr': 1e-2}       # Head (10x)
]
optimizer = torch.optim.SGD(params, momentum=0.9)
```

> 위 코드는 ResNet의 `layer1`~`layer4` 속성명을 그대로 살려 5단계로 나눈 예시입니다. 뒤에 나올 그림과 예시에서는 이를 조금 더 단순화해 Early/Mid/Late/Head 4단계 배율(×0.1/×1/×3/×10)로 표현합니다. 세분화 수준만 다를 뿐, "출력에 가까울수록 학습률을 크게 준다"는 같은 원칙을 보여주는 두 가지 예시입니다.

여기에 **LR Scheduler**를 추가로 얹으면, "층별로 다른 강도" + "시간이 지날수록 점점 약해지는 강도"를 동시에 제어할 수 있습니다.

```python
def get_lr_lambda(epoch):
    if epoch < 10:
        return 1.0
    elif epoch < 20:
        return 0.1
    else:
        return 0.01

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
```

> **실무 팁**: 이 두 기법은 양자택일이 아니라 대부분 함께 씁니다. Parameter Groups가 "공간(층)"을 제어한다면, Scheduler는 "시간(epoch)"을 제어하는 별개의 축이라, 조합했을 때 가장 안정적인 학습이 이뤄집니다. 특히 Full Fine-tuning처럼 리스크가 큰 방식일수록 이 조합의 효과가 커집니다.

### 예시: 단일 학습률 vs 차등 학습률

**가정한 조건 (예시 수치)**: Base LR 0.001, Uniform(모든 층 0.001) vs Differential(Head 0.01 / Late 0.003 / Mid 0.001 / Early 0.0001)

![단일 학습률과 차등 학습률을 비교한 꺾은선 그래프. Differential LR은 Epoch 10에서 81.5%, 20에서 88.7%, 30에서 91.2%를 기록했고, Uniform LR은 각각 75.2%, 83.1%, 86.3%에 그쳤다](/assets/img/posts/transfer-learning-finetuning-guide/04_lr_experiment.svg)

| Approach | Epoch 10 | Epoch 20 | Final | Convergence |
|---|---|---|---|---|
| Uniform | 75.2% | 83.1% | 86.3% | Epoch 25 |
| **Differential** | **81.5%** | **88.7%** | **91.2%** | **Epoch 18** |

이 예시 수치대로라면, 같은 Full Fine-tuning 방식이라도 학습률을 층별로 차등 적용하는 것만으로 **최종 정확도가 4.9%p 향상**되고, **수렴도 7 epoch 더 빨라지는** 효과를 기대할 수 있습니다. 층별 동결이 "많은 층을 학습 대상에서 아예 제외"하는 방법이라면, 차등 학습률은 그 안에서 "학습률 조절"이라는 다른 축을 수행하는 방법입니다.

---

## 3. Gradual Unfreezing: 시간에 따라 점진적으로 풀어주기 {#gradual-unfreezing}

동결 전략(공간 축)과 차등 학습률(시간+공간 축)을 배웠다면, 이 둘을 자연스럽게 이어주는 기법이 **점진적 해제(Gradual Unfreezing)**입니다. 핵심 아이디어는 "Full Freeze로 시작해서, epoch가 지남에 따라 뒤쪽 층부터 순서대로 하나씩 풀어가다 결국 Full Fine-tuning에 도달하는 것"입니다.

![Gradual Unfreezing 타임라인. Epoch 1-5는 Head만 학습, 6-10은 Head+Layer4, 11-15는 Head+L4+L3, 16 이후는 점진적으로 모든 층을 해제한다](/assets/img/posts/transfer-learning-finetuning-guide/05_gradual_unfreeze.svg)

```python
class ProgressiveUnfreeze:
    def __init__(self, model, stages, epochs_per_stage):
        self.model = model
        self.stages = stages  # [layer4, layer3, layer2, layer1] 순서로 unfreeze
        self.epochs_per_stage = epochs_per_stage
        self.current_stage = -1  # 시작은 전부 동결

    def step(self, epoch):
        stage_idx = epoch // self.epochs_per_stage
        if stage_idx > self.current_stage and stage_idx < len(self.stages):
            for param in self.stages[stage_idx].parameters():
                param.requires_grad = True
            self.current_stage = stage_idx
            print(f"Unfroze stage {stage_idx}")

# 사용 예시
unfreeze_scheduler = ProgressiveUnfreeze(
    model,
    [model.layer4, model.layer3, model.layer2, model.layer1],
    epochs_per_stage=5
)
```

이렇게 보면 앞서 "선택지"로 배웠던 3가지 동결 전략이 사실 **하나의 연속적인 학습 여정 위의 순서대로 지나가는 단계들**이라는 걸 알 수 있습니다. 이는 랜덤 초기화된 분류기가 안정될 시간을 주고, 뒤쪽 층(구체적 특징)부터 순서대로 재조정하며, 앞쪽 층(범용 특징)은 최대한 늦게까지 보존합니다.

---

## 4. 데이터 증강 강도: 얼마나 세게 데이터를 흔들 것인가 {#data-augmentation}

모델을 어떻게 학습시킬지(동결/학습률)를 다뤘다면, 이번엔 **학습 데이터 자체를 어떻게 다룰지**입니다. 데이터 증강(Data Augmentation)의 목적은 세 가지입니다.

- **과적합 방지**: 모델이 훈련 데이터를 통째로 암기하는 것을 막음
- **데이터 다양성 증가**: 제한된 데이터로도 다양한 변형을 학습
- **불변성 학습**: 회전, 크기 변화 등에 강인한 모델을 만듦

증강은 보통 강도에 따라 3단계로 나뉩니다.

![데이터 증강 강도 3단계를 나타낸 누적 구조도. Weak는 기본 Flip과 Normalize, Medium은 여기에 RandomResizedCrop·Rotation·ColorJitter를 더하고, Strong은 AutoAugment·RandomErasing·MixUp까지 추가한다](/assets/img/posts/transfer-learning-finetuning-guide/06_augmentation_levels.svg)

```python
from torchvision import transforms
from torchvision.transforms import autoaugment

# Weak: 보수적
weak_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Medium: 균형적
medium_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Strong: 공격적
strong_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    autoaugment.AutoAugment(),
    transforms.RandomErasing(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

Weak → Medium → Strong으로 갈수록 기법이 **누적**되며, crop 범위가 넓어지고 무작위성이 강해집니다.

### 예시: 데이터 크기와 증강 강도의 상호작용

이 예시가 특히 흥미로운데, "어떤 증강이 제일 좋은가"가 아니라 **"데이터 크기에 따라 최적의 증강 강도가 달라진다"**는 걸 보여줍니다.

![데이터셋 크기와 증강 강도별 검증 정확도 히트맵. 500장에서는 Medium이 최고이고, 2,000장과 10,000장에서는 Strong이 최고 성능을 기록한다](/assets/img/posts/transfer-learning-finetuning-guide/07_aug_heatmap.svg)

| Dataset Size | Weak Aug | Medium Aug | Strong Aug |
|---|---|---|---|
| 500 samples | 78.3% | **82.1%** | 79.8% |
| 2,000 samples | 85.7% | 88.9% | **89.5%** |
| 10,000 samples | 91.2% | 92.8% | **94.3%** |

- **작은 데이터셋(500장)**: Medium이 최적입니다. Strong 증강은 원본 정보를 과하게 흐려서 오히려 **underfitting**을 유발합니다.
- **중간 데이터셋(2,000장)**: Strong 증강의 효과가 나타나기 시작하며 과적합이 크게 감소합니다.
- **큰 데이터셋(10,000장)**: Strong 증강으로 최고 성능을 달성하며 일반화 능력이 극대화됩니다.

즉 "동결 전략"과 완전히 같은 논리가 여기서도 반복됩니다 — **데이터가 적을수록 보수적으로, 많을수록 과감하게 건드릴 것.**

### 증강 적용 시 주의사항

1. **분포 왜곡**: 과도한 증강은 원본 데이터 분포를 크게 변형시켜 성능을 오히려 저하시킬 수 있습니다. 의료 영상처럼 미세한 디테일이 중요한 도메인은 특히 유의해야 합니다.
2. **도메인 고려**: 자연 이미지 기준으로 만들어진 증강 기법을 특정 도메인(의료 영상, 위성 영상 등)에 그대로 적용하면 안 됩니다. 예를 들어 흉부 X-ray를 좌우 반전하면 실제로 존재하지 않는 이상 소견을 학습시키는 꼴이 될 수 있습니다.
3. **Validation 세트**: 증강은 학습 데이터에만 적용해야 합니다. 검증/테스트 데이터에 증강을 적용하면 실제 성능을 정확히 평가할 수 없습니다.

---

## 5. 작은 데이터셋을 위한 종합 프로토콜 {#small-dataset-protocol}

지금까지 배운 기법들을 데이터가 매우 적은(클래스당 <100 samples, 전체 <1,000 samples) 가장 까다로운 상황에 맞춰 통합하면, 다음 4가지 핵심 요소로 이루어진 표준 워크플로가 만들어집니다.

| 핵심 요소 | 구체적 방법 |
|---|---|
| 점진적 학습 | 단계별로 층을 해제하며 안정적으로 학습 (Gradual Unfreezing) |
| 강한 정규화 | Dropout, Weight Decay 등 적극 활용 |
| 절제된 증강 | Medium 수준의 균형잡힌 증강 |
| Cross-validation | K-fold로 안정적인 성능 평가 |

### 2단계 학습 프로토콜

```python
# Phase 1: Head Pre-training (Epochs 1-10)
# 전략: 백본 완전 동결
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
# + Medium 증강 적용
# + Dropout 0.5로 강한 정규화

# Phase 2: Gradual Unfreezing (Epochs 11-30)
# 전략: 마지막 stage만 해제
for param in model.layer4.parameters():
    param.requires_grad = True

params = [
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
]
optimizer = torch.optim.Adam(params)
# + 증강 강도 약간 감소
# + Early stopping (patience=5)
```

| | Phase 1 (1-10) | Phase 2 (11-30) |
|---|---|---|
| 학습 대상 | fc만 | fc + layer4 |
| 전략 | 빠르게 분류기를 task에 적응 | 세밀한 task 적응 |
| 종료 조건 | 정해진 10 epoch | Early stopping |

### 추가 정규화 3종 세트

```python
# 1. Dropout — 분류기에 0.5 수준의 강한 드롭아웃
# (모델 정의 시 nn.Dropout(0.5) 추가)

# 2. Weight Decay — L2 정규화로 가중치 크기 제한
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 regularization
)

# 3. Label Smoothing — 과신을 방지하는 부드러운 레이블
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

| 기법 | 무엇을 건드리는가 | 역할 |
|---|---|---|
| Dropout | 뉴런 (모델 구조) | 특정 뉴런 의존도를 낮춰 견고한 표현 학습 |
| Weight Decay | 가중치 크기 (파라미터) | 극단적인 가중치를 억제해 일반화 유도 |
| Label Smoothing | 정답 라벨 (손실 계산) | 100% 확신 대신 약간의 불확실성을 유지 |

---

## 6. 종합: 상황별 전략 선택 가이드 {#decision-guide}

지금까지 다룬 모든 축(동결 전략, 학습률, 증강, 정규화)을 **"데이터 크기 × 도메인 유사도"** 기준으로 정리하면 아래와 같은 하나의 참조표로 요약됩니다.

| 상황 | Freeze Strategy | LR Strategy | Aug Strategy | 추가 팁 |
|---|---|---|---|---|
| Large (>10K) / Similar domain | Partial FT (Last 2 blocks) | Differential Head×10 | Strong | Standard training |
| Large / Different domain | Full FT | Differential + Gradual unfreeze | Strong | Longer training |
| Medium (1K-10K) / Similar domain | Partial FT (Last 1-2 blocks) | Differential Head×5 | Medium-Strong | Careful monitoring |
| Small (<1K) / Similar domain | Head only → Last block | 2-Phase, Head first | Medium | Heavy regularization |
| Small / Different domain | Progressive Unfreeze | 3-Phase Gradual | Medium | Cross-validation |

이 표에서 읽을 수 있는 3가지 일관된 원칙은 다음과 같습니다.

1. **데이터가 많아질수록 더 많이, 더 세게 건드린다.** Freeze Strategy는 Partial → Full로, Aug Strategy는 Medium → Strong으로 강해집니다.
2. **도메인이 다를수록 더 깊이, 더 조심스럽게 건드린다.** 같은 데이터 크기여도 Different domain 쪽은 항상 더 깊게(Full FT, Progressive Unfreeze) 들어가면서도, 동시에 Gradual/Cross-validation 같은 안전 장치가 함께 붙습니다.
3. **위험이 커질수록 안전 장치가 늘어난다.** 데이터가 적어질수록 "추가 팁"이 Standard training → Careful monitoring → Heavy regularization → Cross-validation 순으로 점점 더 신중해집니다.

---

## 핵심 정리

전이 학습 파인튜닝은 결국 하나의 질문으로 요약됩니다.

> **"내가 가진 데이터의 양과 성질을 고려했을 때, 사전 학습된 모델을 얼마나 강하게 건드려야 하는가?"**

이 질문에 답하기 위해 우리는 세 가지 축의 도구를 갖췄습니다.

- **모델을 어떻게 건드릴지**: 동결 전략(Freeze/Partial/Full) + 차등 학습률 + Gradual Unfreezing
- **데이터를 어떻게 다룰지**: 증강 강도(Weak/Medium/Strong)
- **과적합을 어떻게 막을지**: Dropout, Weight Decay, Label Smoothing, Cross-validation

전략은 하나로 고정되어 있지 않습니다. 데이터가 적고 도메인이 비슷하면 최대한 보수적으로, 데이터가 많고 도메인이 다르면 과감하게 — 이 원칙 하나만 기억한다면, 어떤 전이 학습 문제를 만나든 스스로 합리적인 출발점을 잡을 수 있을 것입니다.
