---
title: "CNN 이미지 분류 실습(1) — AlexNet 전이학습으로 개미 vs 벌 분류기 만들기"
date: 2026-09-11 13:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, alexnet, transfer-learning, cnn, image-classification, deep-learning, computer-vision]
description: "AlexNet 사전학습 모델의 마지막 분류 계층만 교체해 개미/벌 이미지를 분류하는 전이학습 실습을 처음부터 끝까지 코드로 따라간다. classifier[6] 교체, 작은 학습률, epoch별 최고 검증 정확도 가중치 저장까지 핵심 로직을 정리했다."
---

> CNN 기반 이미지 분류 수업에서 실습한 노트북들을 정리하는 시리즈입니다. 총 4편에 걸쳐 서로 다른 사전학습 모델과 전략으로 이미지 분류기를 만들어보며, 매번 "왜 이렇게 하는가"에 초점을 맞춥니다. 1편인 이번 글은 **AlexNet + 전이학습**으로 "개미(ant)"와 "벌(bee)" 사진을 구분하는 분류기를 만드는 과정을 다룹니다.

## 왜 전이학습인가?

CNN을 처음부터 학습시키려면 수십만 장의 데이터와 오랜 학습 시간이 필요합니다. 하지만 **전이학습(Transfer Learning)**을 쓰면, 이미 대규모 데이터(ImageNet, 약 120만 장)로 학습된 모델을 가져다가 **마지막 출력층만 내 문제에 맞게 바꿔서** 적은 데이터로도 빠르게 좋은 성능을 낼 수 있습니다.

이번 실습에서 쓰는 **AlexNet**은 2012년 ImageNet 대회에서 우승하며 딥러닝 붐을 일으킨 초기 CNN 모델입니다. 구조는 크게 두 부분으로 나뉩니다.

- **features**: Conv(합성곱) 레이어 5개 + Pooling → 이미지에서 선, 모서리, 질감 등 특징 추출
- **classifier**: Fully Connected 레이어 3개 → 추출된 특징을 보고 최종 클래스 판단 (원래 1000개 클래스)

**features 부분은 그대로 재활용**하고 **classifier의 마지막 레이어(1000개 → 2개 출력)만 새로 교체**하는 것이 전이학습의 핵심입니다.

![AlexNet의 features(Conv 5개 + Pooling)는 그대로 재사용하고, classifier의 마지막 Linear 레이어만 1000개 클래스 출력에서 2개 클래스(개미/벌) 출력으로 교체하는 전이학습 구조도](/assets/img/posts/alexnet/diagram1-transfer-learning.svg)

> **용어 정리**: [지난 글]({% post_url 2026-08-26-pytorch-imagefolder-transfer-learning %})에서 이 블로그는 "파인튜닝 = 모든 파라미터 재학습", "전이학습 = 대부분 동결 후 마지막 계층만 학습"으로 구분했습니다. 이번 노트북은 classifier의 마지막 레이어만 새로 갈아 끼우지만, 뒤에서 보듯 나머지 파라미터를 동결하지 않고 전부 다시 학습시키므로 엄밀히는 **파인튜닝**에 해당합니다. "사전학습 모델을 재사용한다"는 넓은 의미에서 전이학습이라 부르되, 구체적인 학습 방식은 파인튜닝이라는 점을 구분해서 읽어주세요.

---

## 1. 라이브러리 불러오기

```python
import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

torch.manual_seed(0)  # 재현성을 위해 시드 고정
```

## 2. Kaggle 데이터셋 준비

이번 실습에서 쓰는 데이터는 Kaggle의 `hymenoptera-data`(벌목 곤충 데이터셋)로, **개미(ant) / 벌(bee)** 두 클래스 이미지가 들어있습니다.

```python
!pip install kaggle

from google.colab import files
files.upload()  # kaggle.json 업로드

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ajayrana/hymenoptera-data
!unzip -q hymenoptera-data.zip -d .
```

## 3. 이미지 전처리 정의

학습(train)과 검증(val)에 서로 다른 전처리를 적용합니다. 학습 데이터에는 **데이터 증강(augmentation)**을 넣어 모델이 다양한 상황에 강해지도록 합니다.

```python
ddir = 'hymenoptera_data'

data_transformers = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),   # 무작위로 잘라 224x224로 리사이즈
        transforms.RandomHorizontalFlip(),   # 무작위 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),          # 항상 중앙을 224x224로 고정 크롭
        transforms.ToTensor(),
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
    ])
}
```

> **왜 train과 val 전처리가 다를까요?** train은 매번 다른 방식으로 무작위로 잘리고 뒤집혀서, 모델이 "위치나 방향이 조금 달라도 같은 대상"이라는 걸 배우게 도와줍니다(데이터 증강). 반면 val은 성능을 **일관된 기준**으로 평가해야 하므로 무작위성을 없애고 항상 같은 방식(중앙 크롭)으로 처리합니다.

## 4. 데이터셋 & 데이터로더 생성

```python
img_data = {k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k]) for k in {'train', 'val'}}
# ImageFolder: 폴더 이름을 클래스 이름으로 자동 인식 (ants/, bees/ 폴더 구조 활용)

dloaders = {
    'train': torch.utils.data.DataLoader(img_data['train'], batch_size=8, shuffle=True, num_workers=2),
    'val': torch.utils.data.DataLoader(img_data['val'], batch_size=8, shuffle=False, num_workers=2)
}

dset_sizes = {x: len(img_data[x]) for x in {'train', 'val'}}
classes = img_data['train'].classes
dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 5. 이미지 확인용 시각화 함수

정규화된 이미지는 사람 눈으로 보기 이상하기 때문에, 역정규화(denormalization)해서 원래 색깔로 되돌리는 함수를 만듭니다.

```python
def imageshow(img, text=None):
    img = img.numpy().transpose((1, 2, 0))  # (C,H,W) → (H,W,C)

    avg = np.array([0.490, 0.449, 0.411])
    stddev = np.array([0.231, 0.221, 0.230])
    img = stddev * img + avg          # 역정규화: img = stddev * img + avg
    img = np.clip(img, 0, 1)          # [0,1] 범위를 벗어난 값 clip

    plt.imshow(img)
    plt.axis('off')
    if text is not None:
        plt.title(text)
```

```python
d_iter = iter(dloaders['train'])
imgs, cls = next(d_iter)

grid = torchvision.utils.make_grid(imgs)
imageshow(grid, text=[classes[x] for x in cls])
```

## 6. 학습 함수(`finetune_model`) — 이 실습의 핵심 로직

```python
def finetune_model(pretrained_model, loss_func, optim, epochs=10):
    start = time.time()
    model_weights = copy.deepcopy(pretrained_model.state_dict())  # 초기 가중치 백업
    accuracy = 0.0  # 지금까지의 최고 검증 정확도

    for e in range(epochs):
        print(f'epoch_number {e} / {epochs - 1}')
        print('=' * 20)

        for dset in ['train', 'val']:
            if dset == 'train':
                pretrained_model.train()   # Dropout, BatchNorm 활성화
            else:
                pretrained_model.eval()    # Dropout, BatchNorm 비활성화

            loss = 0.0
            successes = 0

            for imgs, tgts in dloaders[dset]:
                imgs = imgs.to(dvc)
                tgts = tgts.to(dvc)

                optim.zero_grad()

                with torch.set_grad_enabled(dset == 'train'):
                    ops = pretrained_model(imgs)
                    _, preds = torch.max(ops, 1)
                    loss_curr = loss_func(ops, tgts)

                    if dset == 'train':
                        loss_curr.backward()
                        optim.step()

                loss += loss_curr.item() * imgs.size(0)
                successes += torch.sum(preds == tgts.data)

            loss_epoch = loss / dset_sizes[dset]
            accuracy_epoch = successes.double() / dset_sizes[dset]
            print(f'{dset} loss in this epoch: {loss_epoch}, accuracy in this epoch: {accuracy_epoch}')

            # 검증 정확도가 지금까지 최고치를 갱신하면 그 시점 가중치를 저장
            if dset == 'val' and accuracy_epoch > accuracy:
                accuracy = accuracy_epoch
                model_weights = copy.deepcopy(pretrained_model.state_dict())
        print()

    time_delta = time.time() - start
    print(f'Training finished in {time_delta // 60}mins {time_delta % 60}secs')
    print(f'Best accuracy: {accuracy}')

    pretrained_model.load_state_dict(model_weights)  # 최고 성능 시점 가중치로 복원
    return pretrained_model
```

**핵심 흐름 요약**: 매 epoch마다 train(학습)과 val(검증)을 번갈아 수행하고, **검증 정확도가 지금까지 중 최고일 때만 그 시점의 가중치를 저장**해둡니다. 학습이 끝나면 저장해둔 최고 성능 가중치로 모델을 복원해서 반환합니다. 마지막 epoch의 가중치가 아니라 "검증 기준 최고 시점"의 가중치를 쓰는 이 패턴은, 뒤에 나올 시리즈의 다른 실습에서도 반복해서 쓰이는 표준 학습 루프입니다.

## 7. 예측 결과 시각화 함수

```python
def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1)
    was_model_training = pretrained_model.training  # 종료 후 원상복구를 위해 저장
    pretrained_model.eval()

    imgs_counter = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(dloaders['val']):
            imgs = imgs.to(dvc)
            tgts = tgts.to(dvc)

            ops = pretrained_model(imgs)
            _, preds = torch.max(ops, 1)

            for j in range(imgs.size()[0]):
                imgs_counter += 1

                ax = plt.subplot(max_num_imgs // 2, 2, imgs_counter)
                ax.axis('off')
                ax.set_title(f'pred: {classes[preds[j]]} || target: {classes[tgts[j]]}')
                imageshow(imgs.cpu().data[j])

                if imgs_counter == max_num_imgs:
                    pretrained_model.train(mode=was_model_training)
                    return
        pretrained_model.train(mode=was_model_training)
```

## 8. AlexNet 불러오고 마지막 레이어 교체하기 ⭐

```python
model_finetuned = models.alexnet(pretrained=True)
print(model_finetuned.classifier)
```

출력해보면 `classifier`는 다음과 같은 구조입니다.

```text
Sequential(
  (0): Dropout(p=0.5)
  (1): Linear(in_features=9216, out_features=4096)
  (2): ReLU(inplace=True)
  (3): Dropout(p=0.5)
  (4): Linear(in_features=4096, out_features=4096)
  (5): ReLU(inplace=True)
  (6): Linear(in_features=4096, out_features=1000)   ← 이걸 교체할 것!
)
```

```python
model_finetuned = models.alexnet(pretrained=True)
# 기존 1,000개 클래스 → 현재 데이터셋 클래스 개수(2개, 벌/개미)로 변경
model_finetuned.classifier[6] = nn.Linear(4096, 2)
```

> **왜 `4096`은 그대로 두고 `1000`만 `2`로 바꿀까요?** `nn.Linear(in_features, out_features)`에서 `in_features=4096`은 바로 앞 레이어가 출력하는 뉴런 개수라서 함부로 바꾸면 텐서 모양이 안 맞아 에러가 납니다. 반면 `out_features`는 "몇 개 클래스로 분류할지"를 결정하는 값이라 우리 데이터(2개 클래스)에 맞게 자유롭게 바꿀 수 있습니다.

## 9. 손실함수 & 옵티마이저 설정 후 학습 실행

```python
loss_func = nn.CrossEntropyLoss()

# 학습률을 매우 작게 설정 — 이미 학습된 모델을 살살 미세조정하기 위함
optim_finetune = optim.SGD(model_finetuned.parameters(), lr=0.0001)

model_finetuned = model_finetuned.to(dvc)
model_finetune = finetune_model(model_finetuned, loss_func, optim_finetune)
```

> `lr=0.0001`처럼 아주 작은 학습률을 쓰는 이유: 이미 ImageNet으로 잘 학습된 가중치를 큰 학습률로 갑자기 크게 흔들면, 애써 배워둔 좋은 특징들이 망가질 수 있습니다(**catastrophic forgetting**). 그래서 미세조정은 작은 학습률로 조심스럽게 진행합니다.
>
> 여기서 `optim_finetune`에 넘긴 `model_finetuned.parameters()`는 새로 교체한 `classifier[6]`뿐 아니라 `features`를 포함한 **전체 파라미터**입니다. 별도로 `requires_grad = False`를 설정해 동결한 부분이 없으므로, 위 용어 정리에서 언급했듯 이 학습 방식은 "전이학습"보다 "파인튜닝"에 더 가깝습니다. 학습률을 아주 작게 잡은 것도 결국 "동결 대신 작은 학습률로 전체를 조심스럽게" 건드리는 전략입니다.

## 10. 최종 결과 확인

```python
visualize_predictions(model_finetune)
```

학습이 끝난 모델로 검증 이미지 몇 장을 예측해보고, 예측(pred)과 정답(target)이 잘 맞는지 눈으로 확인합니다.

---

## 정리

| 단계 | 핵심 내용 |
|---|---|
| 데이터 준비 | Kaggle에서 개미/벌 이미지 다운로드, train/val 전처리 분리 |
| 모델 불러오기 | ImageNet 사전학습 AlexNet 로드 |
| 출력층 교체 | `classifier[6]`만 `nn.Linear(4096, 2)`로 교체 |
| 파인튜닝 | 동결 없이 전체 파라미터를 작은 학습률(`lr=0.0001`)로 10epoch 학습, 최고 검증 정확도 시점 가중치 저장 |
| 결과 확인 | 검증 이미지에 대한 예측 vs 정답 시각화 |

**핵심 교훈**: 사전학습 모델을 재사용하는 것(넓은 의미의 전이학습)은 "이미 똑똑한 모델을 가져다가 내 문제에 맞게 새로 가르치는 것"입니다. 이번 실습처럼 전체 파라미터를 다시 학습시키는 파인튜닝이든, 일부만 동결 해제하는 좁은 의미의 전이학습이든, 둘 다 처음부터 다 학습시키는 것보다 훨씬 적은 데이터와 시간으로 좋은 성능을 낼 수 있습니다.

이 시리즈는 다음 편에서 다른 사전학습 모델과 전략으로 계속됩니다.
