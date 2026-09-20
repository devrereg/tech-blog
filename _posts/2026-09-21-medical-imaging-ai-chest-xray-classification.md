---
title: "의료 영상 분석 AI 완전 정복 — DICOM부터 ResNet18 전이학습, ROC-AUC, Grad-CAM까지"
date: 2026-09-21 09:00:00 +0900
categories: [AI, Deep Learning]
tags: [medical-ai, dicom, chest-xray, pneumonia, resnet, transfer-learning, roc-auc, confusion-matrix, grad-cam, xai, pytorch, deep-learning, computer-vision]
math: true
description: "흉부 X-ray로 폐렴을 판별하는 AI를 DICOM 표준, ResNet18 전이학습, ROC-AUC 평가, Grad-CAM 해석까지 처음부터 끝까지 정리했다. 핵심은 'AI는 의사를 대체하지 않고 신뢰할 수 있게 보조한다'는 원칙이다."
---

> 흉부 X-ray 한 장으로 폐렴 여부를 판별하는 AI를 만든다고 하면 막막하게 느껴질 수 있습니다. 하지만 전체 과정을 쪼개 보면 **데이터를 이해하고 → 모델을 학습시키고 → 제대로 평가하고 → 신뢰할 수 있게 만드는** 네 가지 흐름으로 정리됩니다. 이 글에서는 DICOM 표준 포맷부터 ResNet18 전이학습, ROC-AUC 평가, Grad-CAM 해석까지 의료 영상 분석 AI의 전체 그림을 순서대로 정리합니다.
>
> ResNet 전이학습의 세부 구조는 [전이학습 완전 정복(ResNet·VGG)]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})과 [Fine-tuning 실전 가이드]({% post_url 2026-08-26-transfer-learning-finetuning-guide %})에서, Grad-CAM의 계산 원리는 [설명 가능한 AI(XAI) 완전 정리]({% post_url 2026-09-11-xai-cam-gradcam-saliency-integrated-gradients %})에서 더 깊이 다뤘으니 함께 보면 좋습니다.

## 📌 목차

1. [의료 영상 분석이란](#intro)
2. [DICOM: 의료 영상의 표준 포맷](#dicom)
3. [개발 환경: 무엇으로 만드는가](#environment)
4. [데이터셋: Chest X-Ray Pneumonia](#dataset)
5. [PyTorch 데이터 파이프라인](#pipeline)
6. [ResNet18과 전이학습](#resnet)
7. [모델 학습 프로세스](#training)
8. [모델 평가: 정확도만으로는 부족하다](#evaluation)
9. [ROC Curve와 AUC](#roc-auc)
10. [Precision, Recall, F1-score](#precision-recall)
11. [Grad-CAM: AI의 시선 추적기](#gradcam)
12. [연구 수준으로: 고급 기법과 재현성](#advanced)
13. [마무리](#summary)

---

## 1. 의료 영상 분석이란 {#intro}

의료 영상이란 X-ray, CT, MRI, 초음파처럼 몸 속을 직접 보지 않고도 들여다볼 수 있게 찍은 데이터를 말합니다. 문제는 **양이 너무 많다**는 것입니다. 병원 한 곳에서 하루에도 수백~수천 장의 영상이 쌓이고, 의사가 이 모든 것을 꼼꼼히 살피기엔 시간이 부족합니다.

AI가 필요한 이유는 명확합니다.

- **속도**: 응급 상황(뇌출혈 등)에서 AI는 몇 초 만에 1차 스크리닝을 해줄 수 있습니다.
- **일관성**: 사람은 피곤하면 판단이 흔들리지만, AI는 같은 기준으로 반복 판단합니다.
- **미세 신호 포착**: 아주 작은 초기 병변도 픽셀 단위까지 비교·분석할 수 있습니다.

중요한 전제는, **AI가 의사를 대체하는 게 아니라 숙련된 보조 의사처럼 위험 신호를 먼저 걸러내는 역할**을 한다는 점입니다. 최종 판단과 책임은 여전히 의사에게 있습니다.

실제 구현은 아래 4단계 파이프라인으로 이루어집니다.

![병원에서 X-ray·CT·MRI 영상을 수집하고, 화질 개선과 전문의 라벨링으로 전처리한 뒤, CNN이 수만 장의 영상으로 병변 패턴을 학습하고, 새 영상이 들어오면 AI가 의심 부위를 표시해 의사가 최종 판단하는 4단계 AI 진단 지원 파이프라인](/assets/img/posts/medical-imaging-ai-chest-xray-classification/01-pipeline.svg)

1. **데이터 수집**: 병원에서 X-ray, CT, MRI 영상을 모읍니다.
2. **전처리**: 화질을 다듬고 정답 라벨을 붙입니다 (전문의가 직접 확인).
3. **AI 모델 학습**: CNN 같은 딥러닝 모델에게 수만 장의 영상을 보여주며 패턴을 학습시킵니다.
4. **진단 지원**: 새로운 영상이 들어오면 AI가 의심 부위를 표시하고, 의사가 최종 판단합니다.

---

## 2. DICOM: 의료 영상의 표준 포맷 {#dicom}

의료 영상은 일반 JPG/PNG 파일이 아니라 **DICOM**(Digital Imaging and COmmunications in Medicine)이라는 국제 표준 포맷으로 저장됩니다. 가장 큰 차이는 이것입니다.

> 일반 사진은 "어떻게 생겼는지"만 담지만, DICOM은 "누구를, 언제, 어떤 장비로 찍었는지"까지 한 파일에 통째로 담습니다.

![DICOM 파일이 Header(파일 식별자·버전 정보), Meta Data(환자명·나이·성별·촬영일·병원명·장비 모델), Pixel Data(그레이스케일 또는 컬러 픽셀 값) 세 구획으로 이루어져 있고, Meta Data에 개인정보가 있어 비식별화가 필요함을 보여주는 DICOM 파일 구조도](/assets/img/posts/medical-imaging-ai-chest-xray-classification/02-dicom-structure.svg)

- **Header 정보**: 파일 식별자, 버전 정보 — 파일의 "신분증"
- **Meta Data**: 환자명, 나이, 성별, 촬영 날짜, 병원명, 장비 모델
- **Pixel Data**: 실제 영상 픽셀 값 (그레이스케일 또는 컬러)

Meta Data 덕분에 "60대 여성 환자의 흉부 X-ray만 필터링" 같은 작업이 가능해지지만, 동시에 환자 개인정보가 담겨 있어 AI 학습 전에는 반드시 **비식별화(익명화)** 처리가 필요합니다.

---

## 3. 개발 환경: 무엇으로 만드는가 {#environment}

의료 영상 AI를 실제로 구현할 때 쓰는 핵심 도구들입니다.

| 도구 | 역할 |
|---|---|
| **Python** | AI 개발의 표준 프로그래밍 언어 |
| **PyTorch** | 딥러닝 모델을 만들고 학습시키는 프레임워크 |
| **Google Colab** | 무료 GPU를 브라우저에서 바로 쓸 수 있는 개발 환경 |
| **Kaggle** | 의료 영상 데이터셋을 공유하는 플랫폼 |

라이브러리로는 `torch`/`torchvision`(모델·이미지 처리), `pydicom`(DICOM 파일 읽기), `matplotlib`/`seaborn`(시각화), `grad-cam`(모델 해석), `scikit-plot`(ROC/PR 곡선)을 주로 사용합니다.

한 가지 꼭 기억할 습관은 **재현성 확보**입니다. `torch.manual_seed()`로 랜덤 시드를 고정해야, 같은 코드를 다시 돌려도 항상 같은 결과가 나옵니다. 요리 레시피의 재료 양과 순서를 정확히 지켜야 항상 같은 맛이 나는 것과 같은 원리입니다.

```python
import torch

SEED = 42
torch.manual_seed(SEED)
```

---

## 4. 데이터셋: Chest X-Ray Pneumonia {#dataset}

실습에 흔히 쓰이는 대표 데이터셋은 Kaggle의 **Chest X-Ray Images (Pneumonia)**(Paul Mooney 공개, 광저우 여성·아동 의료센터 수집)이며, 두 클래스로 구성됩니다.

- **NORMAL**: 정상 흉부 X-ray, 약 1,341장
- **PNEUMONIA**: 폐렴 진단 흉부 X-ray, 약 3,875장

![NORMAL 1,341장 대비 PNEUMONIA 3,875장으로 약 2.9배 차이가 나는 클래스 불균형을 보여주는 막대그래프](/assets/img/posts/medical-imaging-ai-chest-xray-classification/03-class-imbalance.svg)

여기서 주목할 점은 **클래스 불균형**입니다. PNEUMONIA가 NORMAL보다 3배 가까이 많습니다. 이걸 그냥 두고 학습시키면, 모델이 "무조건 폐렴"이라고만 찍어도 얼핏 정확도가 높게 나오는 함정이 생깁니다. 대응 방법은 다음과 같습니다.

- **데이터 증강**: 이미지를 회전·반전시켜 적은 클래스의 수를 인위적으로 늘림
- **클래스 가중치 조정**: 적은 클래스의 오답에 더 큰 페널티 부여
- **평가 지표 신경 쓰기**: 단순 정확도 대신 ROC-AUC 같은 지표 사용 (뒤에서 자세히 다룸)

기타 데이터 특성으로는 이미지 포맷이 JPEG(DICOM 변환 완료)이고, 그레이스케일(1채널) 영상이며, 해상도가 제각각이라 전처리가 필요합니다.

> **한계도 함께 알아두어야 합니다.** 이 데이터셋은 소아 환자 위주로 단일 의료기관에서 수집되었고, 라벨링 노이즈 가능성도 보고된 바 있습니다. 즉 여기서 좋은 성능이 나왔다고 해서 다른 병원·연령대·촬영 장비에서도 그대로 재현된다는 보장은 없습니다. 이는 앞서 강조한 "AI는 대체가 아니라 보조"라는 원칙이 데이터 단계에서부터 이미 필요한 이유이기도 합니다.

---

## 5. PyTorch 데이터 파이프라인 {#pipeline}

원본 JPEG 파일이 실제로 모델까지 전달되는 과정은 다음 순서를 따릅니다.

**원본 이미지 → Dataset 클래스 → DataLoader → 모델 학습**

- **Dataset 클래스**: 진열대에서 이미지를 한 장씩 꺼내며 전처리를 적용
- **DataLoader**: Dataset이 꺼낸 이미지를 **배치(예: 32장)** 단위로 묶어 모델에 전달

이미지가 모델에 들어가기 전, `transforms`로 3단계 전처리를 거칩니다.

| 단계 | 하는 일 | 이유 |
|---|---|---|
| **Resize** | 모든 이미지를 224×224로 통일 | 모델이 정해진 크기의 입력만 받기 때문 |
| **ToTensor** | 이미지를 텐서로 변환, 0~1로 정규화 | 컴퓨터는 숫자로만 계산 가능 |
| **Normalize** | 평균·표준편차로 한 번 더 표준화 (ImageNet 통계값) | 전이학습 효과를 제대로 내기 위해 |

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])
```

`ImageFolder`로 데이터셋을 구성하는 실전 코드는 [ImageFolder로 전이학습 실전 적용]({% post_url 2026-08-26-pytorch-imagefolder-transfer-learning %})에서 자세히 다뤘습니다.

---

## 6. ResNet18과 전이학습 {#resnet}

**전이학습(Transfer Learning)**은 처음부터 모델을 새로 학습시키는 대신, 이미 잘 학습된 모델을 가져와 우리 문제에 맞게 살짝만 고쳐 쓰는 방법입니다. 수천 장의 그림을 그려본 화가에게 "이제부터 폐렴 사진만 특별히 잘 그려보라"고 부탁하는 것과 비슷합니다.

- **ImageNet 사전학습**: ResNet18은 이미 1,000개 클래스, 수백만 장의 일반 이미지로 "선을 알아보는 법", "질감을 구분하는 법" 같은 기본기를 갖춘 상태입니다.
- **특징 추출 층은 그대로 재사용**하고, **마지막 FC Layer(출력층)만** 우리 문제(NORMAL/PNEUMONIA 2개 클래스)에 맞게 교체합니다.

FC Layer 교체 수식은 다음과 같습니다.

$$logits = Wx + b, \quad W \in \mathbb{R}^{2 \times d}$$

- **x**: 특징 추출 층을 거쳐 나온 d차원 특징 벡터
- **W**: d개의 특징을 보고 NORMAL/PNEUMONIA 2개 점수를 계산하는 가중치
- **logits**: 최종 산출되는 2개의 점수 (더 높은 쪽이 모델의 판단)

```python
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # NORMAL / PNEUMONIA
```

ResNet과 VGG의 구조적 차이, 특징 추출 층을 얼마나 얼릴지(freeze) 등 더 깊은 내용은 [전이학습 완전 정복(ResNet·VGG)]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})을 참고하세요.

---

## 7. 모델 학습 프로세스 {#training}

한 번의 학습 사이클은 다음 5단계로 이루어지며, 배치마다 반복됩니다.

**① 데이터 로드 → ② 순전파(Forward) → ③ 손실 계산 → ④ 역전파(Backward) → ⑤ 가중치 업데이트 → (반복)**

1. **데이터 로드**: DataLoader가 배치 단위로 이미지와 정답 레이블을 가져옵니다.
2. **순전파(Forward Pass)**: 이미지를 모델에 통과시켜 예측값(logits)을 계산합니다.
3. **손실 계산**: 예측과 정답을 비교해 오차를 숫자로 측정합니다 (`CrossEntropyLoss`).
4. **역전파(Backward Pass)**: 오차가 각 가중치에 얼마나 책임이 있는지 거꾸로 계산합니다 (`loss.backward()`).
5. **가중치 업데이트**: 오차가 줄어드는 방향으로 가중치를 조정합니다 (`optimizer.step()`, Adam 알고리즘).

### CrossEntropyLoss 손실 함수

$$L = -\sum_i y_i \log(p_i)$$

정답 클래스의 예측 확률이 1에 가까울수록(확신을 가지고 맞힐수록) 손실은 0에 가까워지고, 반대로 틀린 확신을 할수록 손실이 커집니다. **정답에 확신을 가질수록 상 받고, 틀린 확신을 가질수록 벌 받는** 채점 기준입니다.

### 주요 하이퍼파라미터

- **Batch size (32)**: 한 번에 처리할 이미지 수
- **Epoch (10~20)**: 전체 데이터를 몇 번 반복할지
- **Learning rate (0.001)**: 가중치를 한 번에 얼마나 크게 수정할지 — 산 정상을 찾아갈 때의 "보폭"에 비유할 수 있습니다. 너무 크면 정상을 지나치고, 너무 작으면 도달이 너무 느립니다.

---

## 8. 모델 평가: 정확도만으로는 부족하다 {#evaluation}

모델이 "폐렴이다/아니다"를 판단한 결과는 실제 정답과 조합해 4가지 경우로 나뉩니다.

![실제 폐렴/정상과 예측 폐렴/정상을 교차한 2x2 혼동 행렬. TP·TN은 정확한 진단(파란색), FP·FN은 오진(주황색)이며 그중 폐렴을 정상으로 오진하는 FN은 치료 시기를 놓쳐 가장 위험하다는 것을 붉은 테두리와 경고 아이콘으로 강조한 그림](/assets/img/posts/medical-imaging-ai-chest-xray-classification/04-confusion-matrix.svg)

- **TP**: 폐렴을 폐렴으로 정확히 진단
- **TN**: 정상을 정상으로 정확히 진단
- **FP (거짓 양성)**: 정상인데 폐렴으로 오진 → 불필요한 추가 검사
- **FN (거짓 음성)**: 폐렴인데 정상으로 오진 → **치료 시기를 놓쳐 가장 위험**

기본 평가 지표인 Accuracy(정확도)는 다음과 같이 계산됩니다.

$$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$

하지만 앞서 본 클래스 불균형 상황에서는 함정이 있습니다. 모델이 무조건 "폐렴"이라고만 찍어도 정확도가 꽤 높게 나올 수 있기 때문입니다. 그래서 **의료 AI에서는 정확도 하나만 보지 않고, 여러 지표를 교차 검증하는 것이 중요합니다.**

---

## 9. ROC Curve와 AUC {#roc-auc}

**ROC Curve**는 분류 기준(threshold)을 바꿔가며 TPR과 FPR이 어떻게 변하는지 그린 그래프입니다.

$$TPR(민감도) = \frac{TP}{TP+FN} \qquad FPR(거짓양성률) = \frac{FP}{FP+TN}$$

- **TPR**: 실제 폐렴 환자 중 몇 %를 놓치지 않고 찾아냈는가 (높을수록 좋음)
- **FPR**: 실제 정상인 중 몇 %를 폐렴으로 오진했는가 (낮을수록 좋음)

판정 기준을 낮추면 TPR과 FPR이 함께 오르고, 기준을 높이면 함께 내려가는 **trade-off** 관계입니다.

![FPR을 x축, TPR을 y축으로 하는 ROC 곡선. 대각선 점선은 무작위 추측(AUC=0.5) 기준선이고, 왼쪽 위 (0,1) 완벽한 분류점에 가깝게 휘어진 파란 곡선과 그 아래 음영으로 표시된 AUC≈0.9(예시) 영역을 보여주는 그래프](/assets/img/posts/medical-imaging-ai-chest-xray-classification/05-roc-curve.svg)

곡선이 **왼쪽 위 모서리**에 가까울수록 좋은 모델이며, 대각선에 가까우면 무작위 추측 수준입니다. **AUC(Area Under the Curve)**는 이 곡선 아래 면적을 숫자 하나로 요약한 값입니다. 위 그림의 AUC≈0.9는 실제 실험값이 아니라 "곡선이 왼쪽 위로 얼마나 휘어지는지"를 보여주기 위한 예시이며, 실제 성능은 데이터 전처리·증강·하이퍼파라미터에 따라 달라집니다.

| AUC 값 | 의미 |
|---|---|
| 0.5 | 동전 던지기 수준 (랜덤) |
| 0.7~0.8 | 괜찮은 성능 |
| 0.9 이상 | 매우 우수한 성능 |
| 1.0 | 완벽한 분류기 |

AUC가 의료 AI에서 특히 선호되는 이유는, **클래스 불균형이 있어도 모델의 전반적인 분류 능력을 공정하게 평가할 수 있기 때문**입니다. TPR·FPR이 각 클래스 내부 비율로 계산되기 때문에 데이터 개수 차이에 흔들리지 않습니다.

---

## 10. Precision, Recall, F1-score {#precision-recall}

$$Precision = \frac{TP}{TP+FP} \qquad Recall = \frac{TP}{TP+FN}$$

- **Recall**: 실제 폐렴 환자 중 몇 %를 잡아냈는가 (TPR과 동일 — "놓치지 않았는가")
- **Precision**: 폐렴이라고 예측한 것 중 몇 %가 진짜였는가 ("false alarm을 얼마나 적게 울렸는가")

**상황에 따라 우선순위가 달라집니다.**

- **High Recall이 중요한 상황**: 암, 폐렴처럼 놓치면 치료 시기를 잃는 스크리닝 단계 — FN을 최소화해야 함
- **High Precision이 중요한 상황**: 수술처럼 큰 부작용이 따르는 침습적 치료 결정 — 확실할 때만 양성 판정해 불필요한 수술 방지

두 지표는 보통 트레이드오프 관계이므로, 이를 하나로 합친 지표가 **F1-score**입니다.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

한쪽만 높고 다른 쪽이 낮으면 F1은 크게 낮아지므로, "어느 한쪽만 잘하는 건 진짜 좋은 모델이 아니다"라는 것을 알려줍니다.

---

## 11. Grad-CAM: AI의 시선 추적기 {#gradcam}

지금까지의 모델은 "폐렴입니다"라는 답은 주지만 **왜** 그렇게 판단했는지는 알 수 없는 블랙박스입니다. **Grad-CAM**(Gradient-weighted Class Activation Mapping)은 이 블랙박스를 열어, 모델이 이미지의 어느 부분을 중요하게 보았는지 히트맵으로 보여주는 **설명 가능한 AI(XAI)** 기법입니다.

![흉부 X-ray를 모델에 입력해 NORMAL 또는 PNEUMONIA로 분류하고, 예측 클래스에 대한 레이어별 기울기를 역전파로 계산해 히트맵을 생성한 뒤 원본 X-ray 위에 겹쳐 시각화하는 5단계 Grad-CAM 처리 과정](/assets/img/posts/medical-imaging-ai-chest-xray-classification/06-gradcam-process.svg)

1. **입력 이미지**: 흉부 X-ray를 모델에 입력
2. **모델 추론**: NORMAL 또는 PNEUMONIA로 분류
3. **기울기 계산**: 예측 클래스에 대한 특정 레이어의 기울기 추출 (역전파 개념 재활용)
4. **히트맵 생성**: 중요 영역을 빨간색(높음)~파란색(낮음)으로 표시
5. **원본 오버레이**: 히트맵을 원본 X-ray 위에 겹쳐서 시각화

의사는 이 히트맵을 보고, AI가 실제 병변 부위를 정확히 짚었는지, 아니면 엉뚱한 부분(잡음 등)을 보고 판단했는지 직접 검토할 수 있습니다. 이는 앞서 강조한 **"AI는 대체가 아니라 보조"**라는 원칙을 기술적으로 실현하는 핵심 도구입니다. 더 정교한 버전으로 ScoreCAM, GradCAM++ 같은 개선된 기법도 있으며, CAM·Saliency Map·Integrated Gradients와의 계산 원리 비교는 [설명 가능한 AI(XAI) 완전 정리]({% post_url 2026-09-11-xai-cam-gradcam-saliency-integrated-gradients %})에서 다뤘습니다.

---

## 12. 연구 수준으로: 고급 기법과 재현성 {#advanced}

실제 연구·현업 수준에서는 다음과 같은 정교한 장치들이 추가됩니다.

- **Train/Validation/Test 분할**: 학습셋으로 가중치를 학습하고, 검증셋(모의고사)으로 하이퍼파라미터를 튜닝하며, 테스트셋(진짜 시험)으로 딱 한 번만 최종 성능을 확인합니다
- **데이터 증강(Data Augmentation)**: RandomRotation, RandomHorizontalFlip 등으로 원본 이미지를 변형해 데이터 다양성을 늘리고 과적합을 방지합니다
- **Early Stopping & LR Scheduler**: 검증 성능이 더 이상 좋아지지 않으면 학습을 자동으로 멈추고, Learning rate를 초반엔 크게 후반엔 작게 조정해 정밀하게 최적점을 찾습니다
- **실험 설정 JSON & 자동 보고서**: 학습 환경, 하이퍼파라미터 전체를 기록하고 결과를 자동으로 보고서화해, 나중에 재현하고 검증할 수 있게 합니다 (재현 가능한 연구, Reproducible Research)

---

## 13. 마무리 {#summary}

> DICOM 데이터를 전처리하고 → ResNet18을 전이학습으로 튜닝하고 → 손실함수와 옵티마이저로 학습 루프를 반복하고 → 정확도가 아닌 다각도 지표(Recall, ROC-AUC, F1)로 평가하고 → Grad-CAM으로 판단 근거를 시각화하고 → 실험을 기록해 재현 가능하게 만든다.

이 흐름이 바로 의료 영상 분석 AI를 개요부터 연구 수준까지 다루는 전체 그림입니다. 핵심은 기술 하나하나보다, **"AI는 의사를 대체하지 않고, 신뢰할 수 있는 방식으로 의사를 보조한다"**는 원칙이 설계 전 과정에 녹아 있다는 점입니다.
