---
title: "CNN 기반 이미지 분류, 제대로 이해하기"
date: 2026-08-20 10:00:00 +0900
categories: [AI, Deep Learning]
tags: [pytorch, deep-learning, cnn, image-classification, computer-vision, machine-learning]
description: "CIFAR-10 이미지 분류 실습을 통해 CNN이 왜 필요한지, 합성곱과 풀링 연산이 어떻게 작동하는지, 완전결합 신경망과 비교해 성능이 얼마나 차이 나는지 직접 확인하며 정리했다."
---

> 이미지 분류 문제를 풀 때 왜 일반적인 신경망(전결합 신경망) 대신 CNN(합성곱 신경망)을 쓰는지, 그리고 파이토치(PyTorch)로 어떻게 구현하는지를 CIFAR-10 데이터셋 실습을 통해 정리합니다.

## 📌 목차

1. [문제 정의: 우리는 왜 CNN이 필요한가](#why-cnn)
2. [CNN의 핵심 구조](#cnn-structure)
3. [합성곱 처리 (Convolutional Layer)](#convolution-layer)
4. [풀링 처리 (Pooling Layer)](#pooling-layer)
5. [1차원 텐서화 (Flatten Layer)](#flatten-layer)
6. [파이토치로 CNN 모델 구현하기](#pytorch-implementation)
7. [완전결합 신경망 vs CNN, 실제로 비교해보기](#fc-vs-cnn)
8. [새로운 과제: 과적합(Overfitting)](#overfitting)
9. [마무리](#summary)

---

## 1. 문제 정의: 우리는 왜 CNN이 필요한가 {#why-cnn}

[지난 글]({% post_url 2026-08-20-pytorch-mnist-multiclass-classification %})에서는 손글씨 숫자(MNIST)를 완전결합 신경망으로 분류해봤습니다. 이번 글에서는 한 단계 더 나아가, 훨씬 복잡한 컬러 이미지(CIFAR-10)를 다룰 때 왜 완전결합 신경망만으로는 한계에 부딪히는지, 그리고 CNN이 이 문제를 어떻게 해결하는지 살펴봅니다.

우리가 눈으로 사물을 볼 때, 단순히 픽셀의 나열을 보는 게 아니라 선, 모서리, 질감 같은 특징들을 조합해서 "이건 고양이구나", "저건 자동차구나" 하고 인식합니다.

**합성곱 신경망(Convolutional Neural Network, CNN)**은 바로 이 방식을 흉내 낸 딥러닝 모델입니다.

기존의 완전결합 신경망(Fully-Connected Network)은 이미지를 길게 펼친 1차원 데이터로 다루다 보니, 픽셀들이 서로 어떻게 이웃하고 있는지 — 즉 이미지의 중요한 **공간적 정보**를 잃어버리는 한계가 있었습니다. CNN은 이미지의 공간 구조를 그대로 유지한 채, 마치 돋보기로 특징을 살펴보듯이 작동하여 이미지 인식을 진행합니다.

## 2. CNN의 핵심 구조 {#cnn-structure}

CNN의 처리 흐름은 크게 3단계로 나눌 수 있습니다.

| 단계 | 이름 | 역할 |
|---|---|---|
| 1 | 특징 추출 (Convolution) | 이미지에서 중요한 패턴과 특징을 찾아냄 |
| 2 | 정보 압축 (Pooling) | 핵심 정보만 남기고 불필요한 세부사항 제거 |
| 3 | 최종 분류 (Fully Connected) | 추출된 특징을 바탕으로 이미지가 무엇인지 판단 |

즉 CNN은 크게 두 부분으로 나뉩니다.

- 이미지의 특징을 추출하는 **특징 추출부(Features)**
- 추출된 특징을 보고 분류하는 **분류부(Classifier)**

![CNN 전체 파이프라인](/assets/img/posts/cnn-basics/cnn_pipeline.svg)

### 레고 조립 전문가 비유

CNN을 레고 조립 전문가에 비유하면 이해하기 쉽습니다.

- **합성곱(Convolution)**: 전문가가 수많은 레고 블록 속에서 "1x2 빨간 블록", "동그란 투명 블록"처럼 특정 모양과 색의 기본 부품(특징)을 찾아내는 단계. 이때 사용하는 "특징을 찾아내는 틀"이 바로 **커널(Kernel)** 입니다.
- **풀링(Pooling)**: 각 구역에서 가장 대표적인 부품 하나씩만 뽑아 전체적인 부품의 수를 줄이는 단계.
- **전결합층(Fully Connected Layer)**: 모인 핵심 부품들의 조합을 보고 "이 부품들이면 자동차를 만들 수 있겠다"고 최종 결론을 내리는 단계.

## 3. 합성곱 처리 (Convolutional Layer) {#convolution-layer}

`nn.Conv2d`의 작동 원리는 다음과 같습니다.

1. 작은 크기의 커널(필터)이 이미지 위를 한 칸씩 이동합니다.
2. 이동할 때마다, 커널이 있는 위치의 이미지 값들과 커널의 값들을 각각 곱한 뒤 모두 더해 결과값 하나를 만듭니다.
3. 이 과정을 이미지 전체에 반복하면 **특징 맵(Feature Map)** 이라는 새로운 결과가 만들어집니다.

말로만 보면 헷갈릴 수 있으니, 아래 위젯에서 커널이 실제로 이미지를 어떻게 스캔하는지 직접 눌러보세요. "다음 칸 이동" 버튼을 누를 때마다 파란 박스(커널)가 한 칸씩 이동하며 곱하고 더하는 계산 과정을 확인할 수 있습니다.

<div style="border: 1px solid #d3d1c7; border-radius: 12px; overflow: hidden; margin: 24px 0;">
  <iframe src="/assets/html/cnn-basics/convolution_kernel_sliding.html" width="100%" height="420" style="border: none; display: block;" title="합성곱 연산 시각화"></iframe>
</div>

### 커널 값 vs 특징 맵 값 (헷갈리기 쉬운 부분)

| 구분 | 역할 | 학습되나요? |
|---|---|---|
| **커널 값** | "어떤 모양을 찾을지" 정하는 도장/틀 | ✅ 학습을 통해 최적화됨 |
| **특징 맵 값** | 도장을 찍어본 결과 (그 위치가 도장 모양과 얼마나 일치하는지) | ❌ 계산 결과일 뿐, 파라미터 아님 |

커널 안의 숫자들(예: `[[1,0,-1],[1,0,-1],[1,0,-1]]`)은 처음에는 랜덤한 값으로 초기화되고, 학습 과정(역전파)을 거치면서 조금씩 업데이트되어 의미 있는 패턴(세로선, 가로선, 색상 패턴 등)을 감지하도록 최적화됩니다.

보통 합성곱 레이어 하나에는 커널이 여러 개 존재합니다. 예를 들어 32개의 커널이 있다면 각기 다른 특징을 감지하는 32장의 특징 맵이 결과로 나옵니다. 이렇게 쌓인 특징 맵의 개수를 **채널(channel)** 이라고 부릅니다.

```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
```

- `in_channels`: 입력 이미지의 채널 수 (RGB면 3, 흑백이면 1)
- `out_channels`: 사용할 커널의 개수 = 출력되는 특징 맵의 장수
- `kernel_size`: 커널의 크기 (3이면 3x3)

레이어를 여러 개 쌓을 때는 **이전 레이어의 `out_channels`가 다음 레이어의 `in_channels`로 그대로 이어져야** 합니다.

```python
nn.Conv2d(in_channels=3,  out_channels=32,  kernel_size=3)  # RGB 입력 → 32채널 출력
nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3)  # 32채널 입력 → 64채널 출력
```

> **참고: 왜 출력 크기가 줄어들까?**
> 패딩(padding)을 따로 설정하지 않으면 출력 크기는 `입력 크기 - 커널 크기 + 1`로 계산됩니다 (커널이 한 칸씩 이동하는 **stride=1** 기준). 3x3 커널을 쓰면 항상 가로·세로가 2씩 줄어듭니다. 커널이 이미지 가장자리를 완전히 벗어날 수 없기 때문에, 커널을 놓을 수 있는 시작 위치의 개수 자체가 줄어드는 것이 원인입니다. stride를 2 이상으로 늘리면 커널이 건너뛰며 이동하기 때문에 출력 크기가 더 작아지는데, 이 글에서는 다루지 않습니다.

## 4. 풀링 처리 (Pooling Layer) {#pooling-layer}

`nn.MaxPool2d`는 합성곱을 거친 특징 맵의 크기를 줄여주는 과정입니다. 대표적인 방식인 **최대 풀링(Max Pooling)**은 특정 구역(예: 2x2)에서 가장 큰 값만 남기고 나머지는 버립니다.

풀링을 쓰면 크게 두 가지 효과를 얻습니다.

- **크기 축소**: 특징 맵의 크기를 줄여 계산량 감소
- **강인함(robustness) 확보**: 이미지 내에서 특징의 위치가 약간 바뀌어도 대응할 수 있는 능력

아래 위젯에서 "다음 구역 이동" 버튼을 눌러보면, 4x4 특징 맵을 2x2씩 나눠 각 구역의 최댓값만 뽑아 2x2 결과로 압축하는 과정을 확인할 수 있습니다.

<div style="border: 1px solid #d3d1c7; border-radius: 12px; overflow: hidden; margin: 24px 0;">
  <iframe src="/assets/html/cnn-basics/max_pooling_2x2.html" width="100%" height="380" style="border: none; display: block;" title="최대 풀링 연산 시각화"></iframe>
</div>

**합성곱과 풀링의 차이**

| 구분 | 합성곱 | 풀링 |
|---|---|---|
| 학습되는 파라미터 | ✅ 있음 (커널 값) | ❌ 없음 |
| 목적 | 특징을 찾아냄 | 정보를 압축·요약 |
| 크기 변화 | 커널 크기만큼 소폭 감소 | 보통 절반으로 축소 (2x2 기준) |

## 5. 1차원 텐서화 (Flatten Layer) {#flatten-layer}

합성곱과 풀링을 거친 데이터는 여전히 `[채널, 세로, 가로]` 정보를 가진 다차원 텐서입니다. 하지만 최종 분류를 담당하는 `nn.Linear`는 1차원 벡터 형태의 입력만 받습니다.

`nn.Flatten`은 이 다차원 텐서를 1차원으로 펼쳐주는 다리 역할을 합니다. 이때 **배치(batch) 차원은 그대로 두고, 나머지(채널·세로·가로) 차원만 곱해서 하나로 합칩니다.**

```
Flatten 이전: [100, 32, 14, 14]   ← 4차원 텐서 (배치 100, 채널 32, 세로 14, 가로 14)
Flatten 적용: 32 × 14 × 14 = 6272
Flatten 이후: [100, 6272]         ← 2차원 텐서
```

배치 차원을 건드리지 않는 이유는, 100장의 이미지가 서로 독립적인 데이터라서 절대 섞이면 안 되기 때문입니다. 나중에 손실(loss)을 계산할 때 각 이미지의 예측 결과와 정답을 1:1로 대응시켜야 합니다.

> **CNN도 결국 펼치지 않나요?**
> 맞습니다. 다만 차이가 있습니다. 완전결합 신경망은 **처음부터** 원본 이미지를 1차원으로 펼쳐 공간 정보를 초반에 잃어버리지만, CNN은 합성곱과 풀링으로 **충분히 특징을 추출한 뒤**, 분류 직전에만 1차원으로 펼칩니다. "펼치는 것" 자체가 문제가 아니라 "언제 펼치느냐"가 핵심입니다.

## 6. 파이토치로 CNN 모델 구현하기 {#pytorch-implementation}

지금까지 배운 레이어들을 조합해 실제 CNN 클래스를 설계하면 다음과 같습니다.

```python
class CNN(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)     # 입력 채널:3, 출력 채널:32, 커널:3x3
        self.conv2 = nn.Conv2d(32, 32, 3)    # 입력 채널:32, 출력 채널:32, 커널:3x3
        self.relu = nn.ReLU(inplace=True)    # 활성화 함수 ReLU
        self.maxpool = nn.MaxPool2d((2, 2))  # 풀링 영역 크기: 2x2
        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(6272, n_hidden)      # 입력: 6272, 출력: 은닉 노드 수
        self.l2 = nn.Linear(n_hidden, n_output)  # 입력: 은닉 노드 수, 출력: 클래스 수

        # 역할별로 그룹화
        self.features = nn.Sequential(
            self.conv1, self.relu, self.conv2, self.relu, self.maxpool
        )
        self.classifier = nn.Sequential(self.l1, self.relu, self.l2)

    def forward(self, x):
        x1 = self.features(x)     # 특징 추출
        x2 = self.flatten(x1)     # 1차원으로 펼치기
        x3 = self.classifier(x2)  # 최종 분류
        return x3
```

### 핵심 개념 두 가지

**ReLU (활성화 함수)**: 음수는 0으로, 양수는 그대로 유지하는 단순한 함수입니다. 합성곱은 곱하고 더하는 선형 연산이라, 선형 연산만 계속 쌓으면 층을 아무리 깊게 쌓아도 결국 하나의 선형 연산과 다르지 않습니다. ReLU 같은 비선형 함수를 중간에 넣어야 복잡한 패턴을 학습할 수 있습니다.
>
> 위 코드에서는 `self.relu` 인스턴스 하나를 `features`와 `classifier`에서 반복해서 재사용합니다. ReLU는 학습되는 파라미터가 없는 순수 계산 함수이기 때문에, 여러 위치에서 같은 인스턴스를 공유해도 안전합니다(레이어마다 새로 만들 필요가 없습니다). 반면 `nn.Conv2d`나 `nn.Linear`처럼 학습 파라미터(가중치)를 갖는 레이어는 절대 재사용하면 안 됩니다 — 서로 다른 위치가 같은 가중치를 공유하게 되어버립니다.

**nn.Sequential**: 여러 레이어를 순서대로 나열만 하면, 앞 레이어의 출력이 자동으로 다음 레이어의 입력으로 연결됩니다. 코드가 간결해질 뿐 아니라, `features`(특징 추출부)와 `classifier`(분류부)라는 역할 구분이 코드 구조 자체에 드러난다는 장점이 있습니다.

### Flatten 직전 크기는 어떻게 계산할까?

CNN 설계에서 가장 헷갈리는 부분이 바로 `nn.Linear`에 들어갈 입력 크기(6272) 계산입니다. 입력 이미지부터 순서대로 shape을 손으로 추적하면 다음과 같습니다.

```
Input:   [100, 3, 32, 32]     # 100개 이미지, 3채널(RGB), 32x32 크기
              ↓ conv1 = nn.Conv2d(3, 32, 3)
Conv1:   [100, 32, 30, 30]    # 채널 3→32, 크기 32→30 (32-3+1)
              ↓ relu (shape 변화 없음)
              ↓ conv2 = nn.Conv2d(32, 32, 3)
Conv2:   [100, 32, 28, 28]    # 채널 유지, 크기 30→28 (30-3+1)
              ↓ maxpool = nn.MaxPool2d((2, 2))
MaxPool: [100, 32, 14, 14]    # 채널 유지, 크기 절반으로 축소
              ↓ flatten
Flatten: [100, 6272]          # 32 × 14 × 14 = 6272
```

**핵심 관찰 두 가지**

1. ReLU 활성화 함수는 값만 바꾸므로 shape은 변하지 않습니다.
2. 2x2 풀링을 거치면 이미지의 가로·세로 크기가 정확히 절반으로 줄어듭니다.

## 7. 완전결합 신경망 vs CNN, 실제로 비교해보기 {#fc-vs-cnn}

같은 CIFAR-10 데이터셋(10개 클래스, 32x32 컬러 이미지), 같은 손실 함수(`nn.CrossEntropyLoss`), 같은 옵티마이저(`SGD`, `lr=0.01`) 조건에서 모델 구조만 다르게 하여 학습시켰습니다.

```python
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```

두 모델 모두 아래와 동일한 형태의 학습 루프로 훈련시키고, 매 에폭마다 검증 정확도를 기록했습니다. (`net`에는 완전결합 신경망 또는 CNN을 바꿔 끼워 넣습니다.)

```python
epochs = 50
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(epochs):
    # --- 훈련 ---
    net.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # --- 검증 ---
    net.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    history["train_loss"].append(running_loss / len(train_loader))
    history["val_loss"].append(val_loss / len(val_loader))
    history["val_acc"].append(correct / total)

    print(f"Epoch {epoch+1}/{epochs}, Val Acc: {correct/total:.3f}")
```

- `net.train()` / `net.eval()`: 드롭아웃·배치정규화처럼 훈련·평가 시 동작이 달라지는 레이어를 위해 모드를 명시적으로 전환합니다. 지금 모델에는 해당 레이어가 없지만, 습관적으로 붙여두는 것이 안전합니다.
- `torch.no_grad()`: 검증 단계는 가중치를 업데이트하지 않으므로 기울기 계산을 꺼서 메모리와 연산을 아낍니다.
- `history` 딕셔너리에 에폭별 지표를 쌓아두면, 학습이 끝난 뒤 정확도·손실 곡선을 그려 과적합 여부를 눈으로 확인할 수 있습니다 (8장 참고).

### 완전결합 신경망 결과

CIFAR-10 이미지를 `view(-1)`로 1차원(3072개 숫자)으로 펼친 뒤 완전결합층에 입력했습니다.

- 초기 정확도: **37.7%**
- 최종 정확도(50 에폭 후): **53.0%**

검증 정확도는 약 30 에폭 부근에서 정체되었습니다. 10개 클래스를 무작위로 찍을 때의 정확도(10%)보다는 높지만, 실용적으로 쓰기엔 아쉬운 성능입니다.

### CNN 결과

같은 이미지를 3차원 구조(`[3, 32, 32]`) 그대로 유지한 채 CNN에 입력했습니다.

- 초기 정확도: **34.6%**
- 최종 정확도(50 에폭 후): **66.3%**

### 최종 비교

| | 완전결합 신경망 | CNN |
|---|---|---|
| 최종 정확도 | 53% | **66% (+13%p)** |
| 원인/특징 | 공간 정보 손실 | 공간 구조 유지 |
| 학습 패턴 | 30 에폭에서 정체 | 20 에폭 이후 과적합 |

이미지의 공간 정보를 활용한 CNN이 전결합형 모델보다 **13%p** 더 높은 성능을 보였습니다. 같은 데이터, 같은 학습 조건에서도 오직 모델 구조의 차이만으로 이런 성능 차이가 발생했다는 점이 핵심입니다.

## 8. 새로운 과제: 과적합(Overfitting) {#overfitting}

CNN의 학습 곡선을 보면 새로운 문제가 나타납니다. 훈련 정확도는 거의 100%에 근접할 정도로 치솟지만, 검증 정확도는 20 에폭 근처에서 66% 부근에 도달한 뒤 더 오르지 않고, 검증 손실은 오히려 다시 증가하기 시작합니다.

![CNN 학습 곡선: 훈련 vs 검증 정확도·손실 그래프](/assets/img/posts/cnn-basics/overfitting_learning_curve.png)
_왼쪽: 훈련 정확도는 계속 상승하지만 검증 정확도는 20 에폭 이후 66% 부근에서 정체됩니다. 오른쪽: 훈련 손실은 계속 감소하는 반면 검증 손실은 같은 지점부터 다시 증가합니다 — 두 곡선이 벌어지기 시작하는 지점이 바로 과적합의 신호입니다._

이 현상을 **과적합(Overfitting)** 이라고 부릅니다. 모델이 일반적인 패턴을 배우는 대신, 훈련 데이터를 통째로 "암기"해버리는 현상입니다.

실용적인 해결책은 검증 성능이 정체되거나 나빠지기 시작하는 시점(약 20 에폭)에서 학습을 멈추는 것입니다. 이를 **조기 종료(Early Stopping)** 라고 합니다.

> CNN이 더 강력한 만큼, 훈련 데이터를 더 잘 암기할 수 있는 능력도 커져서 과적합이라는 위험이 함께 따라옵니다. 좋은 모델을 쓸수록 "언제 학습을 멈출지"도 함께 고민해야 한다는 실전 교훈을 얻을 수 있습니다.

## 9. 마무리 {#summary}

오늘 정리한 내용을 한 문장으로 요약하면 다음과 같습니다.

> 이미지는 공간 정보가 중요한 데이터이고, CNN은 그 공간 정보를 유지하며 특징을 추출하기 때문에 완전결합 신경망보다 이미지 분류에 훨씬 효과적이다.

**전체 흐름 정리**

```
이미지 (공간 정보 유지)
   ↓ 합성곱 (Conv) — 특징 추출, 커널이 학습 파라미터
   ↓ ReLU — 비선형성 부여 (shape 불변)
   ↓ 풀링 (Pooling) — 크기 압축, 파라미터 없음
   ↓ (Conv + ReLU + Pool 반복)
   ↓ Flatten — 1차원으로 펼침
   ↓ 전결합층 (Linear) — 최종 분류
```

다음 단계로는 과적합을 완화하는 다른 기법들(드롭아웃, 데이터 증강, 배치 정규화 등)이나, 더 유명한 CNN 아키텍처(ResNet, VGG 등)를 살펴보는 것을 추천합니다.

---

*이 글은 PyTorch 기반 CNN 이미지 분류 수업 자료를 정리한 것입니다. 코드 예제는 실제 동작을 위해 데이터 경로, 모델 하이퍼파라미터 등을 환경에 맞게 조정해야 할 수 있습니다.*
