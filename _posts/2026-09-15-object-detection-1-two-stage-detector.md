---
title: "객체 검출(1) — Two-Stage Detector 완전 정리: R-CNN에서 Faster R-CNN까지"
date: 2026-09-15 00:30:00 +0900
categories: [AI, Deep Learning]
tags: [object-detection, r-cnn, fast-rcnn, faster-rcnn, rpn, selective-search, iou, nms, map, computer-vision, deep-learning]
math: true
description: "R-CNN → Fast R-CNN → Faster R-CNN까지, 두 단계로 객체를 찾는 딥러닝 검출기의 발전사를 IoU·NMS·mAP 같은 평가 도구와 함께 처음부터 끝까지 정리했다."
---

> Classification과 Localization을 넘어, 이미지 안의 여러 객체를 동시에 찾아내는 **Object Detection**을 다루는 시리즈입니다. 1편인 이번 글은 정확도를 최우선으로 하는 **Two-Stage Detector** — R-CNN → Fast R-CNN → Faster R-CNN의 발전 과정을 처음부터 끝까지 정리합니다. 속도를 우선하는 One-Stage Detector(YOLO 계열)는 다음 편에서 다룹니다.
>
> [CNN 완전 정복]({% post_url 2026-08-28-cnn-convolution-complete-guide %})에서 다룬 합성곱(Convolution)과 특징맵(Feature Map) 개념이, 이 글에서 다루는 CNN 기반 검출기 전체의 전제가 됩니다.

## 📌 목차

1. [컴퓨터 비전의 4대 과제](#four-tasks)
2. [객체 검출이 어려운 이유](#why-hard)
3. [초기 접근: Sliding Window의 실패](#sliding-window)
4. [Region Proposal: Selective Search](#selective-search)
5. [검출 성능의 언어: IoU, NMS, mAP](#metrics)
6. [Two-Stage Detector란?](#two-stage)
7. [R-CNN (2014)](#rcnn)
8. [Fast R-CNN (2015)](#fast-rcnn)
9. [Faster R-CNN (2015)](#faster-rcnn)
10. [전체 발전사 한눈에 보기](#summary-table)
11. [핵심 요약](#summary)

---

## 1. 컴퓨터 비전의 4대 과제 {#four-tasks}

객체 검출을 이해하려면 먼저 컴퓨터 비전 분야에서 다루는 과제들이 난이도 순으로 어떻게 발전해왔는지 알아야 합니다.

![Classification, Localization, Object Detection, Instance Segmentation 4개 카드를 난이도 순으로 나란히 비교하는 다이어그램. Classification은 강아지 사진에 라벨 1개만, Localization은 라벨과 박스 1개, Object Detection은 강아지와 고양이 각각에 박스와 라벨, Instance Segmentation은 픽셀 단위 마스크를 출력으로 표시한다](/assets/img/posts/two_stage_detection/01_cv_tasks.svg)

| Task | 질문 | 출력 형태 |
|---|---|---|
| **Classification** | 뭐가 있는가? | 클래스 라벨 1개 |
| **Localization** | 어디 있는가? (객체 1개 가정) | 라벨 + bounding box 1개 |
| **Object Detection** | 여러 개가 각각 어디에? | 여러 개의 (라벨 + box) |
| **Instance Segmentation** | 픽셀 단위로 정확히 어디? | 픽셀 단위 마스크 |

Object Detection을 정의하면 **"두 개 이상의 object들에 대해 Classification + Localization을 동시에 수행하는 것"** 입니다. 이 글에서는 이를 구현하는 여러 접근 방식 중에서도 **Two-Stage Detector**가 어떻게 동작하는지 다룹니다.

### Classification + Localization의 기본 구조

객체가 1개라고 가정했을 때, 딥러닝 모델은 다음처럼 동작합니다.

```
이미지 → CNN (Feature Extraction)
              ├─→ FC Layer → Classification (Dog=0.9, Cat=0.1)
              └─→ FC Layer → BBox Regression (x, y, w, h)
```

하나의 CNN으로 특징을 뽑은 뒤, **두 개의 FC Layer(head)** 로 갈라져서 "무엇인지"와 "어디 있는지"를 동시에 예측합니다. 좌표값 x, y, w, h는 연속적인 숫자이기 때문에 분류(Classification)가 아니라 **회귀(Regression)** 로 학습합니다.

---

## 2. 객체 검출이 어려운 이유 {#why-hard}

Object Detection이 Classification보다 훨씬 어려운 이유는 두 갈래로 정리할 수 있습니다.

**기술적 복잡성**
- 분류와 위치 예측을 동시에 정확히 처리해야 하는 이중 과제
- 크기·형태·각도가 제각각인 객체들을 일관되게 검출해야 함
- 자율주행 등에서는 실시간 처리 속도까지 요구됨

**데이터와 환경**
- 이미지 대부분이 배경이라 객체와 배경을 구분하기 어려움
- 객체마다 위치+클래스를 표시해야 해서 annotation 비용이 큼
- 객체끼리 겹치거나 일부만 보이는 가림(occlusion) 현상

---

## 3. 초기 접근: Sliding Window의 실패 {#sliding-window}

가장 단순하게 떠올릴 수 있는 방법은 **Sliding Window** 입니다.

- 다양한 크기·비율의 윈도우를 이미지 전체에 이동시키면서 매번 "여기 객체가 있는가?"를 CNN으로 검사
- 문제: 이미지 대부분이 배경인데도 **모든 위치, 모든 크기**를 다 검사해야 해서 계산량이 기하급수적으로 증가
- 결과: 실시간 처리가 사실상 불가능

> 무식하게 다 훑어보는 방식은 정확할 수 있어도 너무 느려서 실용적이지 않습니다.

이 문제를 해결하기 위해 등장한 것이 **Region Proposal**, 즉 "어디를 봐야 할지 먼저 똑똑하게 후보를 추려내자"는 전략입니다.

---

## 4. Region Proposal: Selective Search {#selective-search}

**Selective Search**는 객체가 있을 가능성이 높은 후보 영역(Region Proposal)만 추출해서 검사 효율을 획기적으로 높이는 알고리즘입니다. 모든 영역을 검사하는 대신 이미지당 **약 2,000개**의 유망한 영역만 선택합니다.

핵심 아이디어는 단순합니다: **"객체는 비슷한 특성을 가진 픽셀들이 모여 있다"**

### 동작 과정 4단계

1. **초기 Over-segmentation**: pixel intensity 기반 graph 알고리즘으로 이미지를 아주 잘게 쪼갬
2. **초기 Region Proposal 등록**: 쪼개진 모든 조각을 bounding box로 변환해 리스트에 추가
3. **유사 Segment 그룹화**: 컬러·무늬·크기·형태가 비슷한 인접 조각들을 반복적으로 병합
4. **반복**: 병합될 때마다 새로운 크기의 box를 계속 리스트에 추가, 더 합칠 게 없을 때까지 반복

이렇게 **모든 병합 단계의 결과를 누적**하기 때문에, 작은 객체부터 큰 객체까지 다양한 크기를 놓치지 않고 후보에 포함시킬 수 있습니다 (= 높은 Recall).

> ⚠️ **주의**: Selective Search 자체도 느립니다. 반복적인 병합 연산이 CPU에서 이루어지기 때문에 이미지 한 장당 **약 2초**가 걸립니다. 이 "2초"라는 병목이 이후 R-CNN 계열 모델의 발전 과정에서 계속 등장하는 핵심 화두입니다.

---

## 5. 검출 성능의 언어: IoU, NMS, mAP {#metrics}

R-CNN 계열 모델 구조로 들어가기 전에, 검출 결과를 "얼마나 잘했는지" 측정하는 평가 도구들을 먼저 알아야 합니다. 이 도구들은 R-CNN, Fast R-CNN, Faster R-CNN 어디에서나 공통으로 사용됩니다. 특히 IoU는 결과를 사후에 "평가"하는 용도로 그치지 않고, 9절에서 볼 RPN이 anchor 하나하나를 학습 데이터의 positive/negative로 나누는 기준으로도 그대로 재사용됩니다.

### 5.1 IoU (Intersection over Union)

모델이 예측한 박스와 실제 정답(Ground Truth)이 얼마나 겹치는지 나타내는 지표입니다.

$$IoU = \frac{\text{교집합 영역(Area of Overlap)}}{\text{합집합 영역(Area of Union)}}$$

![초록색 Ground Truth 박스와 파란색 Prediction 박스가 대각선으로 겹쳐 있고, 겹치는 부분(교집합)이 노란색으로 강조되어 있으며, 우측에는 IoU = 교집합/합집합 공식과 PASCAL VOC·COCO의 판정 기준, gIoU 보완 개념을 정리한 박스가 있는 다이어그램](/assets/img/posts/two_stage_detection/03_iou.svg)

PASCAL VOC Challenge에서는 **IoU ≥ 0.5** 이면 예측 성공으로 판단합니다. MS COCO는 훨씬 엄격해서 **IoU 0.5부터 0.95까지 0.05 간격**으로 평가한 값을 평균 냅니다. 그래서 같은 모델이라도 COCO 기준 mAP가 PASCAL VOC 기준보다 낮게 나옵니다.

**gIoU (Generalized IoU)**: 두 박스가 전혀 겹치지 않으면 IoU는 항상 0이 되어 "조금 빗나갔는지, 완전히 멀리 떨어졌는지" 구분할 수 없습니다. gIoU는 두 박스를 감싸는 최소 박스 C를 도입해 이 문제를 보완합니다.

$$gIoU = IoU - \frac{|C \setminus (A \cup B)|}{|C|}$$

IoU가 둘 다 0이어도, 박스가 멀리 떨어질수록 gIoU는 더 낮은 음수(최소 -1)로 내려가서 "얼마나 틀렸는지" 방향성 있는 신호를 줍니다.

### 5.2 NMS (Non-Maximum Suppression)

Object Detection 알고리즘은 객체 하나 주변에 비슷한 위치의 박스를 **여러 개 중복 생성**하는 경향이 있습니다. NMS는 이 중복 박스들을 정리해서 가장 적합한 박스 하나만 남기는 후처리 기법입니다.

![Before 패널에는 강아지 한 마리 주변에 confidence 0.93, 0.74, 0.61, 0.58인 박스 4개가 겹쳐 있고, NMS 화살표를 거친 After 패널에는 confidence 0.93인 박스 하나만 남아 있는 전후 비교 다이어그램](/assets/img/posts/two_stage_detection/04_nms.svg)

**동작 알고리즘 (클래스별로 수행)**

1. Confidence threshold(예: 0.5) 이하의 낮은 신뢰도 박스를 먼저 제거
2. 남은 박스를 confidence 높은 순으로 정렬
3. 가장 높은 confidence 박스를 기준으로 선택하고, 이 박스와 **IoU가 threshold(예: 0.4) 이상**인 박스들을 모두 "중복"으로 간주해 제거
4. 남은 박스에 대해 2~3을 반복, 더 이상 처리할 박스가 없을 때까지 진행

> 💡 핵심은 "이미지 전체에서 1등만 남기기"가 아니라, **"겹치는 그룹 안에서만 1등을 남기고, 겹치지 않는(=다른 객체일 가능성이 있는) 박스는 독립적으로 살려둔다"** 는 점입니다. IoU가 0인 두 박스(서로 다른 위치)는 각각 자기 그룹 내에서 따로 NMS가 적용됩니다.

Confidence threshold와 IoU threshold를 조정하면 결과가 달라집니다.

- Confidence threshold를 **높이면** → 검출 수 감소 (더 확실한 것만 남김)
- IoU threshold를 **낮추면** → "조금만 겹쳐도 중복"으로 간주되어 더 많은 박스가 제거됨

### 5.3 Precision, Recall, mAP

**Confusion Matrix로 본 4가지 케이스**

| | 예측 Negative | 예측 Positive |
|---|---|---|
| **실제 Negative** | TN | FP (오탐) |
| **실제 Positive** | FN (놓침) | TP (정확한 검출) |

Object Detection에서 FP는 세 가지 원인으로 나뉩니다: 클래스를 잘못 예측(wrong class), IoU가 threshold 미달(IoU < 0.5), 아예 엉뚱한 곳에 박스(no overlap).

$$Precision = \frac{TP}{TP + FP} \qquad Recall = \frac{TP}{TP + FN}$$

- **Precision**: "내가 있다고 예측한 것 중 진짜 맞은 비율" — 확실한 것만 골라 예측하면 쉽게 높일 수 있음
- **Recall**: "실제 존재하는 것 중 놓치지 않고 찾은 비율" — 모든 후보를 다 Positive로 예측하면 쉽게 높일 수 있음

두 지표는 서로 **트레이드오프** 관계입니다. 어느 쪽이 더 중요한지는 도메인에 따라 다릅니다.

- **의료 영상 판별**: 환자를 놓치면(FN) 위험이 크므로 **Recall이 우선**
- **스팸 메일 판별**: 정상 메일을 스팸으로 오판(FP)하면 위험이 크므로 **Precision이 우선**

Confidence threshold를 조정하면 이 둘의 균형이 바뀝니다. threshold를 낮추면 더 많은 박스가 통과해 Recall은 오르지만 Precision은 떨어지고, threshold를 높이면 반대가 됩니다.

**Precision-Recall Curve와 mAP**

Confidence threshold를 점점 낮춰가며 그때마다 Precision·Recall을 계산해 점을 찍으면 Precision-Recall Curve가 만들어집니다. 이 곡선 아래 면적이 한 클래스에 대한 **AP(Average Precision)** 이고, 모든 클래스의 AP를 평균 낸 것이 **mAP(mean Average Precision)** 입니다.

![Recall을 x축, Precision을 y축으로 하는 곡선이 왼쪽 위(1.0)에서 시작해 오른쪽 아래(0.15 부근)로 완만히 감소하고, 곡선 아래 영역이 옅은 파란색으로 채워져 AP(곡선 아래 면적)를 나타내는 Precision-Recall Curve 그래프. 우측에는 "1개 클래스 = AP, 모든 클래스 AP 평균 = mAP"라는 설명 박스가 있다](/assets/img/posts/two_stage_detection/05_pr_curve_map.svg)

> **mAP 값이 높을수록** 모든 클래스에 대해 정확하고(Precision) 빠짐없이(Recall) 검출한다는 뜻입니다.

---

## 6. Two-Stage Detector란? {#two-stage}

> Two-Stage Detector는 객체 검출을 두 단계로 나누어 수행합니다.
> 1) 첫 번째 단계에서 **Region Proposal**을 생성
> 2) 두 번째 단계에서 각 영역을 **분류하고 위치를 정제**

이 방식은 **정확도를 최우선**으로 하며, 지금부터 살펴볼 **R-CNN 계열**이 대표적입니다. (참고로 두 단계를 하나로 합쳐 한 번에 처리하는 방식은 One-Stage Detector, 예: YOLO — 이건 다음 편 주제입니다.)

지금까지 배운 도구들이 이 파이프라인 안에서 각자의 역할을 맡습니다.

| 개념 | Two-Stage Detector에서의 역할 |
|---|---|
| Selective Search | 1단계: 후보 영역 생성 |
| Classification + BBox Regression | 2단계: 분류 + 위치 정제 |
| IoU | positive/negative 샘플 구분, 학습·평가 기준 |
| NMS | 최종 출력 전 중복 박스 정리 |
| mAP | 모델 전체 성능 평가 |

이제 R-CNN → Fast R-CNN → Faster R-CNN이 이 파이프라인을 어떻게 구체화하고 개선해왔는지 살펴보겠습니다.

![R-CNN, Fast R-CNN, Faster R-CNN 세 파이프라인을 세로 단계로 나란히 비교하는 다이어그램. R-CNN 열은 Selective Search부터 2,000개 region 각각 CNN 실행, 별도 SVM 분류, BBox Regression까지가 모두 빨간색으로 표시되어 병목·분리학습 문제를 나타낸다. Fast R-CNN 열은 전체 이미지 CNN 1회(파란색)와 ROI Pooling(파란색)을 도입했지만 Selective Search(빨간색)가 그대로 남아있다. Faster R-CNN 열은 Selective Search 자리가 초록색 RPN으로 대체되어 있다](/assets/img/posts/two_stage_detection/02_pipelines.svg)

---

## 7. R-CNN (2014): 딥러닝 기반 검출의 시작점 {#rcnn}

### 처리 파이프라인

1. **Selective Search**: 이미지당 약 2,000개의 region proposal 생성
2. **CNN Feature 추출**: 각 region을 고정 크기로 warp하여 **AlexNet**에 통과시켜 특징 추출
3. **SVM 분류**: 추출된 feature로 객체 클래스 판별 (CNN이 아니라 별도의 SVM이 담당!)
4. **Bounding Box Regression**: 예측 박스(P)를 Ground Truth(G)에 맞게 이동·크기 조정하는 변환을 학습. 절대 좌표가 아니라 **박스 크기에 비례한 상대적인 offset** $(t_x, t_y, t_w, t_h)$ 을 예측하며 L2 정규화로 과적합을 방지

$$t_x = \frac{G_x - P_x}{P_w} \qquad t_y = \frac{G_y - P_y}{P_h} \qquad t_w = \log\frac{G_w}{P_w} \qquad t_h = \log\frac{G_h}{P_h}$$

중심 좌표 이동량 $(t_x, t_y)$ 은 박스의 너비·높이로 정규화하고, 크기 변화 $(t_w, t_h)$ 은 로그 비율로 표현합니다. 이렇게 하면 박스 크기가 제각각이어도 같은 스케일의 값으로 회귀할 수 있어 학습이 안정적입니다. 이 파라미터화는 R-CNN뿐 아니라 뒤에서 볼 **RPN의 anchor regression에도 그대로 재사용**됩니다.

### 핵심 문제점

- **극도로 느린 속도**: 2,000개 region 각각에 대해 CNN을 실행하여 이미지당 **47초** 소요
- **Multi-stage Training**: CNN, SVM, Regressor를 따로 학습해야 하는 복잡한 과정
- **메모리 부담**: 모든 region의 feature를 디스크에 저장해야 함
- **End-to-end 불가**: SVM과 Regressor가 CNN feature를 업데이트하지 못함 (학습 결과가 CNN에 피드백되지 않음)

그럼에도 R-CNN은 **"CNN으로 뽑은 특징이 객체 검출에도 효과적이다"** 는 것을 처음 증명한, 딥러닝 기반 검출의 선구적 연구로 평가받습니다.

---

## 8. Fast R-CNN (2015): ROI Pooling 혁신 {#fast-rcnn}

### 핵심 개선: 전체 이미지를 CNN에 1회만 통과

R-CNN의 가장 큰 비효율은 2,000개 region 각각에 대해 CNN을 실행하는 것이었습니다. Fast R-CNN은 **전체 이미지를 CNN에 한 번만 통과**시켜 feature map을 얻은 뒤, 이 **공유된 feature map**에서 각 region의 feature를 추출합니다.

```
R-CNN:       후보를 먼저 자르고 → 각각 CNN에 넣기 (2,000번 반복)
Fast R-CNN:  이미지 전체를 먼저 CNN에 넣고 → feature map에서 후보 위치만 잘라내기 (1번)
```

### 5단계 파이프라인

1. **Full Image CNN**: 전체 이미지를 CNN에 통과시켜 feature map 생성 (1회)
2. **Region Proposal**: 여전히 **Selective Search**로 후보 영역 추출 (원본 이미지 기준)
3. **ROI Projection**: 원본 이미지 좌표의 region proposal을 feature map 좌표로 투영(변환)
4. **ROI Pooling**: 각 region을 고정 크기(예: 7×7)의 feature로 변환
5. **Classification + Regression**: FC layer로 분류와 박스 좌표 조정을 동시에 수행 (SVM 불필요, CNN 내부에서 처리)

> 중요: Region Proposal(후보를 찾는 것)과 ROI Pooling(찾은 후보를 고정 크기로 만드는 것)은 서로 다른 단계입니다. Fast R-CNN에서도 **후보를 찾는 방법 자체는 R-CNN과 동일하게 Selective Search**를 사용합니다. 달라진 것은 CNN을 몇 번 돌리는지와, 후보 영역을 어떻게 고정 크기로 만드는지(ROI Pooling)입니다.

### ROI Pooling 원리

서로 다른 크기의 region feature를 고정된 H×W 크기(예: 7×7)로 통일합니다. Feature map을 grid로 분할하고 각 cell에서 **max pooling**을 수행하여, 입력 크기와 무관하게 일정한 출력을 생성합니다. (채널 수는 그대로 유지됩니다. 예: 7×7×512)

이 고정 크기 출력 덕분에, 크기가 제각각인 후보 영역들도 동일한 크기의 입력이 필요한 FC Layer에 넣을 수 있게 됩니다.

### 성과

| 지표 | 개선 내용 |
|---|---|
| Training 속도 | **8.8배 향상** (84시간 → 9.5시간) |
| Testing 속도 | **146배 향상** (47초 → 0.32초) |
| mAP | 66.0% → **66.9%** (속도뿐 아니라 정확도도 소폭 상승) |

※ 위 수치는 PASCAL VOC 2007 test 기준입니다. 이후 나올 mAP 수치도 모두 같은 기준입니다.

속도만 빨라진 게 아니라 **정확도도 함께 오른 이유**는 학습 방식 자체가 바뀌었기 때문입니다. R-CNN은 CNN(feature 추출) → SVM(분류) → Regressor가 서로 분리되어 학습되기 때문에, CNN이 뽑는 feature가 검출에 최적인지 보장할 수 없었습니다. Fast R-CNN은 분류 loss와 회귀 loss를 하나로 묶어 CNN까지 역전파하는 **multi-task loss**로 학습하기 때문에, CNN 자체가 검출에 도움이 되는 방향으로 feature를 뽑도록 유도됩니다.

### 남은 병목

> **Selective Search가 여전히 CPU에서 이미지당 2초씩 소요**되어 전체 파이프라인의 속도를 제한합니다.

CNN 연산(0.32초)은 매우 빨라졌지만, 이와 별개로 동작하는 Selective Search의 2초가 병목으로 남아있습니다. 이 문제를 해결하는 것이 Faster R-CNN의 과제가 됩니다.

---

## 9. Faster R-CNN (2015): 완전한 End-to-End {#faster-rcnn}

### 핵심 혁신: RPN (Region Proposal Network)

Faster R-CNN의 핵심 혁신은 Selective Search를 **학습 가능한 신경망(RPN)**으로 대체한 것입니다. 이로써 전체 검출 파이프라인이 하나의 통합된 네트워크가 되어 **end-to-end 학습**이 가능해졌습니다.

RPN은 한 마디로, **"후보 영역을 찾는 작업 자체를 CNN으로 학습시키자"**는 아이디어입니다.

| | Selective Search | RPN |
|---|---|---|
| 정체 | 전통적 알고리즘(색상·질감 분석) | 신경망(CNN 기반) |
| 실행 위치 | CPU | GPU (feature map 위에서) |
| 속도 | 이미지당 ~2초 | 훨씬 빠름 (밀리초 단위) |
| 학습 가능 여부 | ❌ 고정된 규칙 | ✅ 데이터로 계속 개선됨 |

### Anchor Box 개념

Feature map의 **각 위치마다** 미리 정의된 여러 개의 참조 박스(anchor)를 배치합니다. Anchor는 검출하려는 객체의 초기 템플릿 역할을 하며, RPN은 이를 정제하여 정확한 proposal을 생성합니다.

- **3가지 크기**: 128², 256², 512² 픽셀
- **3가지 비율**: 1:1, 1:2, 2:1
- **총 9개 조합**: 각 위치에서 다양한 크기의 객체를 검출 가능

### Anchor를 Positive/Negative로 나누기

RPN이 각 anchor를 "객체(foreground)"인지 "배경(background)"인지 학습하려면, 먼저 학습 데이터에서 각 anchor에 정답 라벨을 붙여야 합니다. 이때 기준이 되는 것이 바로 5.1절에서 다룬 **IoU**입니다.

- Ground Truth와 **IoU > 0.7**인 anchor → **Positive** (객체)
- 모든 Ground Truth와 **IoU < 0.3**인 anchor → **Negative** (배경)
- 그 사이(0.3 ≤ IoU ≤ 0.7)인 anchor → 애매하므로 학습에서 **제외**

한 이미지에는 anchor가 (feature map 위치 수) × 9개, 보통 수만 개가 생기는데, 이 중 배경(negative)이 압도적으로 많습니다. 그대로 학습하면 배경 쪽으로 치우치므로, Positive·Negative를 합쳐 보통 **256개** 정도로 샘플링해 균형을 맞춘 뒤 학습에 사용합니다.

![왼쪽 패널에는 feature map 격자 위 한 칸을 중심으로 크기(소형·중형·대형)와 비율(1:1, 2:1, 1:2)이 다른 9개의 anchor box가 겹쳐 그려져 있다. 오른쪽 패널에는 Feature Map이 3×3 Conv Sliding Window를 거친 뒤 Classification(2×9=18 출력, 객체/배경 판단)과 Regression(4×9=36 출력, (x,y,w,h) offset) 두 갈래로 갈라지는 RPN 헤드 구조도가 있다](/assets/img/posts/two_stage_detection/07_anchor_rpn.svg)

### RPN 구조

Feature map에서 **3×3 conv로 sliding window**를 수행하고, 각 위치의 9개 anchor에 대해 두 갈래로 예측합니다.

- **Classification (2개 출력 × 9)**: 객체 있음 / 배경 — "무슨 클래스인지"는 아직 따지지 않음
- **Regression (4개 출력 × 9)**: (x, y, w, h)의 offset — anchor를 정답에 가깝게 조정

같은 이름의 "Sliding Window"이지만, 예전(3절)처럼 원본 이미지에서 매번 CNN 전체를 돌리는 게 아니라 **이미 계산된 작은 feature map 위에서 3×3 conv 하나만 슬라이딩**하기 때문에 매우 가볍습니다.

### Anchor에서 Proposal로: NMS로 추리기

RPN은 한 이미지당 수만 개의 anchor 각각에 대해 "객체일 확률(objectness score)"과 offset을 예측하지만, 이 전부를 다음 단계(Fast R-CNN 헤드)로 그대로 넘기면 R-CNN 초기의 비효율이 되풀이됩니다. 그래서 RPN 출력에도 **NMS**를 적용해 후보를 추립니다.

1. objectness score 기준으로 anchor(정확히는 offset이 적용된 proposal)를 정렬
2. 상위 N개(학습 시 약 12,000개, 추론 시 약 6,000개)만 남기고 나머지는 버림
3. 남은 후보에 **NMS**(IoU threshold 0.7)를 적용해 중복 제거
4. 최종적으로 상위 **2,000개(학습) / 300개(추론)** 만 Region Proposal로 확정해 Fast R-CNN 헤드로 전달

즉 anchor(수만 개) → objectness 상위 N개 → NMS → 최종 proposal(수백~수천 개) 순으로 좁혀지는 것이며, 이 단계에서도 4절·5.2절에서 배운 **NMS**가 그대로 재사용됩니다.

### Joint Training: 4가지 Loss 동시 최적화

RPN과 Fast R-CNN을 함께 학습하여 다음 4가지 loss를 동시에 최적화합니다.

1. RPN classification loss (anchor 품질)
2. RPN regression loss (anchor → proposal)
3. Fast R-CNN classification loss (클래스 분류)
4. Fast R-CNN regression loss (proposal → bbox)

### 성과

| 지표 | 결과 |
|---|---|
| Testing 시간 | **0.2초** (Fast R-CNN 대비 10배 개선) |
| mAP | **66.9% 유지** (속도와 정확도 모두 확보) |
| End-to-End | **100%** (완전한 통합 학습 가능) |

※ 마찬가지로 PASCAL VOC 2007 test 기준입니다.

속도가 10배 빨라졌는데도 정확도는 떨어지지 않았습니다. 오히려 RPN이 anchor를 학습하면서 proposal 품질 자체가 Selective Search보다 나아졌기 때문입니다.

---

## 10. 전체 발전사 한눈에 보기 {#summary-table}

![처리 시간(로그 스케일)과 mAP를 나란히 비교하는 막대그래프 2개. 왼쪽 그래프에서 R-CNN은 47초로 가장 높은 막대, Fast R-CNN은 2.3초, Faster R-CNN은 0.2초로 급격히 낮아진다. 오른쪽 그래프에서는 세 모델의 mAP가 66.0%, 66.9%, 66.9%로 거의 동일한 높이를 유지한다](/assets/img/posts/two_stage_detection/06_speed_map.svg)

| | R-CNN (2014) | Fast R-CNN (2015) | Faster R-CNN (2015) |
|---|---|---|---|
| Region Proposal | Selective Search | Selective Search (여전히 병목) | **RPN** (CNN 내부, GPU) |
| CNN 실행 횟수 | 2,000회 | **1회** | 1회 |
| 분류 방식 | 별도 SVM | CNN 내부 FC layer | CNN 내부 FC layer |
| 처리 시간/이미지 | 47초 | 0.32초 + SS 2초 ≈ 2.3초 | **0.2초** |
| 학습 방식 | Multi-stage (분리 학습) | 부분적 End-to-end | **완전한 End-to-end** |

R-CNN(47초)에서 Faster R-CNN(0.2초)까지, **정확도(PASCAL VOC 2007 mAP)는 유지·향상하면서 속도는 약 235배 개선**되었습니다.

```
R-CNN            문제: 2,000번 CNN 실행 → 너무 느림
   ↓
Fast R-CNN       해결: CNN 1회 실행(ROI Pooling) / 남은 문제: Selective Search 2초
   ↓
Faster R-CNN     해결: Selective Search → RPN(신경망)으로 대체 / 완전한 End-to-End
```

---

## 11. 핵심 요약 {#summary}

**Two-Stage 철학**
Region Proposal과 Classification을 분리하여 정확도를 최우선으로 추구합니다. 실시간성보다 정밀한 검출이 중요한 의료 영상, 보안 시스템 등에 적합합니다.

**R-CNN의 진화**
R-CNN → Fast R-CNN → Faster R-CNN으로 발전하며 속도는 수백 배 개선되었지만 정확도는 유지 또는 향상되었습니다. **ROI Pooling**과 **RPN**이 핵심 혁신입니다.

**평가의 중요성**
IoU, NMS, mAP는 객체 검출 모델을 이해하고 개선하는 데 필수적인 개념입니다. 각 지표의 의미와 조정 방법(confidence threshold, IoU threshold)을 정확히 알아야 실무에서 활용할 수 있습니다.

**핵심 체크리스트**

- [ ] Object Detection은 여러 객체에 대한 Classification + Localization의 동시 수행이다
- [ ] Selective Search는 유사 픽셀을 반복 병합해 약 2,000개의 region proposal을 생성하지만 CPU에서 이미지당 2초가 걸린다
- [ ] IoU는 예측 박스와 정답 박스의 겹침 정도이며, PASCAL VOC는 0.5, COCO는 0.5~0.95 평균을 기준으로 삼는다
- [ ] NMS는 confidence 최고 박스를 기준으로 IoU가 높은 중복 박스만 제거하고, 겹치지 않는 박스는 독립적으로 남긴다
- [ ] mAP는 클래스별 Precision-Recall Curve 아래 면적(AP)을 모든 클래스에 대해 평균한 값이다
- [ ] R-CNN은 2,000개 region 각각에 CNN을 돌려 이미지당 47초가 걸리고, 분리 학습으로 end-to-end가 불가능하다
- [ ] Fast R-CNN은 CNN을 1회만 실행하고 ROI Pooling으로 후보를 고정 크기로 만들지만, Selective Search 2초는 그대로 남는다
- [ ] Faster R-CNN은 Selective Search를 RPN(anchor 9개 기반 신경망)으로 대체해 완전한 end-to-end 학습을 달성한다
- [ ] RPN은 Ground Truth와의 IoU가 0.7 초과면 positive, 0.3 미만이면 negative로 anchor 라벨을 정해 학습한다
- [ ] RPN도 자체적으로 objectness 상위 선별 + NMS를 거쳐 수만 개의 anchor를 수백~수천 개의 proposal로 줄인 뒤 Fast R-CNN 헤드로 넘긴다

---

## 다음 편 예고

이 글에서 다룬 Two-Stage Detector는 **정확도를 우선시하지만 구조가 복잡하고 속도가 상대적으로 느린** 접근입니다. 다음 편에서는 Region Proposal과 Classification을 아예 하나의 단계로 합쳐버리는 **One-Stage Detector(YOLO 계열)** 를 다룹니다. Two-Stage가 "신중하게 두 번 보는" 방식이라면, One-Stage는 "한 번에 빠르게 보는" 방식이라는 점에서 흥미로운 대조를 이룹니다.

*이 글은 R-CNN, Fast R-CNN, Faster R-CNN의 원리와 발전 과정을 IoU·NMS·mAP 같은 평가 도구와 함께 정리한 학습 노트입니다.*
