---
title: "CNN 이미지 분류 실습(2) — VGG13, 미세조정 없이 사전학습 모델 그대로 써보기"
date: 2026-09-11 14:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, vgg13, transfer-learning, cnn, image-classification, deep-learning, computer-vision, imagenet]
description: "ImageNet 사전학습 VGG13을 아무 수정 없이 개미/벌 사진에 그대로 돌려보며, classifier를 건드리지 않은 모델이 왜 우리 문제를 못 푸는지 관찰한다. imagenet1000_clsidx_to_labels.txt를 dict로 읽어 1000개 클래스 이름을 매핑하는 과정까지 정리했다."
---

> [지난 편]({% post_url 2026-09-11-cnn-classification-1-alexnet-transfer-learning %})에서는 AlexNet의 `classifier[6]`를 교체하고 학습시켜 개미/벌을 구분하는 분류기를 만들었습니다. 이번 편은 반대로 **모델을 전혀 수정하지 않고**, ImageNet으로 학습된 VGG13을 그대로 가져와 우리 사진(개미/벌)에 돌려보면서 "왜 전이학습·파인튜닝이 필요한지"를 직접 체감해보는 실습입니다.

## VGG13이란?

VGG는 2014년 옥스포드 연구팀이 만든 CNN 계열 모델입니다. 핵심 아이디어는 **"큰 필터 하나 대신 작은 3×3 필터를 여러 겹 쌓자"**는 것입니다. AlexNet은 11×11, 5×5 같은 큰 필터를 썼지만, VGG는 3×3 필터만 반복해서 층을 깊게(deep) 쌓았습니다. 층이 깊어질수록 더 복잡한 패턴을 인식할 수 있습니다.

- **VGG13**: 가중치를 가진 레이어가 13개(Conv 10개 + FC 3개)
- AlexNet보다 깊고, 구조가 단순·규칙적이라 이해하기 쉬움

이번 실습에서는 VGG13의 분류기를 **건드리지 않고**, 원래 ImageNet 1000개 클래스 그대로 사용합니다.

![VGG13을 수정 없이 그대로 불러와 개미/벌 이미지를 입력하면, 학습되지 않은 모델은 여전히 ImageNet의 1000개 클래스 중에서만 답을 고르기 때문에 개미/벌이 아닌 엉뚱한 곤충 이름을 예측하게 되는 흐름도](/assets/img/posts/vgg13/diagram2-vgg13-inference.svg)

---

## 1. 구글 드라이브 연결

```python
from google.colab import drive
drive.mount('/content/drive')
# /content/drive/MyDrive/ 경로로 내 구글 드라이브 파일에 접근할 수 있게 됨
```

Colab에서 구글 드라이브를 연결합니다. 뒤에서 드라이브 안에 있는 **ImageNet 클래스 이름 파일**을 읽어오기 위한 준비입니다.

## 2. 라이브러리 불러오기

```python
import os          # 파일 경로 조작 (os.path.join 등)
import time        # 시간 측정
import copy         # 객체 깊은 복사 (모델 가중치 백업 등)

import numpy as np              # 배열 연산, 이미지 역정규화에 사용
import matplotlib.pyplot as plt # 이미지 시각화

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

torch.manual_seed(0)  # 재현성을 위해 시드 고정
```

## 3. VGG13 모델 불러오기

```python
model = models.vgg13(pretrained=True)
```

ImageNet으로 미리 학습된 VGG13 가중치를 그대로 불러옵니다. `pretrained=True`라서 다운로드하자마자 **1000개 클래스를 분류할 줄 아는 완성된 모델**이 준비됩니다. 지난 편의 `model_finetuned.classifier[6] = nn.Linear(4096, 2)`에 해당하는 코드가 이번 실습에는 **없습니다** — 이게 이번 편의 핵심입니다.

## 4. Kaggle 데이터셋 준비 (지난 편과 동일한 패턴)

```python
from google.colab import files
files.upload()  # kaggle.json 업로드

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ajayrana/hymenoptera-data
!unzip -q hymenoptera-data.zip -d .
```

개미/벌 이미지 데이터셋을 받습니다. 이번엔 이 데이터로 **학습을 시키지 않고**, 그냥 "모델한테 보여줄 이미지"로만 사용합니다.

## 5. 전처리 + 데이터로더 구성

```python
ddir = 'hymenoptera_data'

# data_transformers: {'train': Compose(...), 'val': Compose(...)} 형태의 딕셔너리.
# 실제로 이번 실습에서 학습은 하지 않지만, ImageFolder에는 전처리가 필수라 두 세트 모두 정의함.
data_transformers = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),   # 무작위로 잘라 224x224 이미지로 리사이즈 (아직 PIL 이미지, 텐서 변환 전)
        transforms.RandomHorizontalFlip(),   # 무작위 좌우 반전
        transforms.ToTensor(),               # PIL 이미지 (H,W,C) → 텐서 (C,H,W), 값 범위 [0,1]
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])  # 채널(R,G,B)별 평균/표준편차로 정규화
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),              # 짧은 변을 256으로 맞춰 리사이즈
        transforms.CenterCrop(224),           # 중앙 224x224만 잘라냄 (증강 없이 항상 동일)
        transforms.ToTensor(),
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
    ])
}

# img_data: {'train': ImageFolder 객체, 'val': ImageFolder 객체}
# 각 ImageFolder는 (이미지 텐서, 정수 라벨) 쌍을 인덱스로 꺼낼 수 있는 Dataset.
# 라벨은 하위 폴더 이름(ants, bees)을 알파벳 순으로 정렬해 0, 1로 자동 매핑됨.
img_data = {k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k]) for k in ['train', 'val']}

# dloaders: {'train': DataLoader, 'val': DataLoader}
# 각 DataLoader를 순회하면 (imgs, tgts) 튜플이 나옴.
#   imgs: shape (batch_size, 3, 224, 224) 의 float 텐서
#   tgts: shape (batch_size,) 의 정수 텐서, 값은 0(ants) 또는 1(bees)
dloaders = {
    'train': torch.utils.data.DataLoader(img_data['train'], batch_size=8, shuffle=True, num_workers=2),
    'val': torch.utils.data.DataLoader(img_data['val'], batch_size=8, shuffle=False, num_workers=2)
}

# dset_sizes: {'train': 244, 'val': 153} — 각 데이터셋의 전체 이미지 장수
dset_sizes = {x: len(img_data[x]) for x in {'train', 'val'}}

# classes: ['ants', 'bees'] — img_data['train']의 폴더명 기반 라벨 목록.
# 주의: 이번 실습에서는 이 2개짜리 리스트를 예측 결과 표시에 쓰지 않는다 (6절 참고).
classes = img_data['train'].classes

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 6. ImageNet 클래스 이름 불러오기 ⭐ (이번 실습만의 특징)

```python
import ast

with open('/content/drive/MyDrive/data/imagenet1000_clsidx_to_labels.txt') as f:
    classes_data = f.read()
# classes_data: 파일 내용 전체가 담긴 하나의 문자열.
# 내용 형태는 파이썬 딕셔너리 리터럴과 똑같이 생겼음 → "{0: 'tench', 1: 'goldfish', ..., 999: 'toilet tissue'}"

classes_dict = ast.literal_eval(classes_data)
# classes_dict: {int: str} 형태의 진짜 파이썬 딕셔너리. 키는 0~999 클래스 인덱스, 값은 클래스 이름 문자열.
# ast.literal_eval()은 문자열 속 파이썬 리터럴(dict, list, 숫자, 문자열 등)만 안전하게 평가해준다.
# eval()과 달리 임의 코드 실행 위험이 없어 "신뢰할 수 없는 문자열을 자료구조로 변환"할 때 권장되는 방법.

print({k: classes_dict[k] for k in list(classes_dict)[:5]})
# 출력 예: {0: 'tench', 1: 'goldfish', 2: 'great white shark', 3: 'tiger shark', 4: 'hammerhead'}
```

이번 실습만의 핵심 차이점입니다. AlexNet 실습에서는 `img_data['train'].classes`로 **개미/벌** 두 개 이름만 썼지만, 이번엔 모델을 수정하지 않았기 때문에 여전히 **1000개 클래스**를 예측합니다. 그래서 "0번은 무슨 동물, 1번은 무슨 물건..." 식으로 1000개 클래스 이름이 담긴 파일을 따로 읽어와야 합니다. 즉 이 실습에는 클래스 이름 목록이 **두 개** 공존합니다 — `classes`(개미/벌, 2개, 실제로는 안 씀)와 `classes_dict`(ImageNet, 1000개, 예측 결과 출력에 사용).

## 7. 시각화 함수 정의

```python
def imageshow(img, text=None):
    # img: 입력은 (C, H, W) 텐서 → numpy 변환 후 (H, W, C)로 축 순서를 바꿔야 matplotlib이 이해함
    img = img.numpy().transpose((1, 2, 0))

    avg = np.array([0.490, 0.449, 0.411])
    stddev = np.array([0.231, 0.221, 0.230])
    img = stddev * img + avg          # 정규화의 역연산: (정규화값 × 표준편차) + 평균 = 원래 픽셀값
    img = np.clip(img, 0, 1)          # 부동소수점 오차로 [0,1]을 살짝 벗어난 값 방지

    plt.imshow(img)
    plt.axis('off')
    if text is not None:
        plt.title(text)


def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1)
    was_model_training = pretrained_model.training  # 함수 종료 후 원래 모드로 복구하기 위해 저장
    pretrained_model.eval()  # Dropout/BatchNorm을 추론 모드로 고정

    imgs_counter = 0
    fig = plt.figure()

    with torch.no_grad():  # 추론만 할 것이므로 gradient 계산 비활성화 (메모리/속도 이득)
        for i, (imgs, tgts) in enumerate(dloaders['val']):
            # imgs: (batch_size, 3, 224, 224) — 이 배치엔 최대 8장의 이미지가 들어있음
            # tgts: (batch_size,) — 개미/벌 정답 라벨이지만, 이번 실습에서는 아래에서 쓰지 않음
            imgs = imgs.to(dvc)

            ops = pretrained_model(imgs)
            # ops: (batch_size, 1000) — 이미지 한 장당 ImageNet 1000개 클래스 각각에 대한 점수(logit)

            _, preds = torch.max(ops, 1)
            # preds: (batch_size,) — 각 이미지마다 1000개 점수 중 가장 높은 값의 "인덱스"(0~999)

            for j in range(imgs.size()[0]):
                imgs_counter += 1
                ax = plt.subplot(max_num_imgs // 2, 2, imgs_counter)
                ax.axis('off')

                # preds[j]는 0~999 사이의 텐서 스칼라 → int로 바꿔 classes_dict의 키로 사용
                ax.set_title(f'pred: {classes_dict[int(preds[j])]}')
                imageshow(imgs.cpu().data[j])

                if imgs_counter == max_num_imgs:
                    pretrained_model.train(mode=was_model_training)
                    return
        pretrained_model.train(mode=was_model_training)
```

> **왜 정답(target)이 표시되지 않을까요?** 우리 데이터는 "개미/벌" 2종류인데, 모델은 ImageNet 1000종류 중에서 답을 고릅니다. "정답=ant"와 "예측=grasshopper" 같은 결과는 애초에 같은 기준으로 비교가 안 됩니다. 이 실습은 정확도를 재는 게 아니라, **"학습 안 시킨 모델이 우리 사진을 보고 뭐라고 착각하는지" 관찰**하는 게 목적입니다.

## 8. 모델을 device로 이동

```python
model = model.to(dvc)
```

## 9. 실행!

```python
visualize_predictions(model)
```

개미/벌 사진 4장을 VGG13에 넣고, 모델이 뭐라고 판단하는지 확인합니다. 아마 결과는 "ant"나 "bee"가 아니라 ImageNet에 있는 비슷한 곤충 이름(예: grasshopper, mosquito, dragonfly 등)이 나올 가능성이 높습니다.

---

## 지난 편(AlexNet)과의 핵심 차이

| 구분 | ① AlexNet 실습 | ② VGG13 실습 (이번 편) |
|---|---|---|
| 목적 | 사전학습 모델을 개미/벌 문제에 맞게 재학습 (지난 편 기준으로는 파인튜닝) | 사전학습 모델 그대로 추론만 |
| classifier 수정 | O (`classifier[6]`: 1000→2) | X (그대로 1000개) |
| 학습(`finetune_model`) | O (10 epoch 학습) | X (학습 안 함) |
| 사용한 클래스 이름 | `classes` — ant, bee (2개) | `classes_dict` — ImageNet 1000개 이름 |
| 목적하는 결과 | 개미/벌을 정확히 구분 | "안 배운 모델이 뭐라고 착각하는지" 관찰 |

## 정리

이번 실습은 **"전이학습·파인튜닝을 왜 해야 하는지"를 체험시켜주는 비교 실습**입니다. 아무리 좋은 사전학습 모델이라도, 내가 풀려는 문제(개미 vs 벌, 2개 클래스)에 맞게 마지막 레이어를 교체하고 다시 학습시켜야 실제로 쓸 만한 분류기가 됩니다. 그렇지 않으면 모델은 여전히 "자기가 원래 알던 1000개 클래스" 중에서만 답을 고르려고 하기 때문입니다.

이 시리즈는 다음 편에서 CNN을 밑바닥부터 직접 쌓아보는 실습으로 계속됩니다.
