---
title: "객체 검출(2) — One-Stage Detector 완전 정복: YOLO v5 → v8 → v10 진화"
date: 2026-09-20 21:00:00 +0900
categories: [AI, Deep Learning]
tags: [object-detection, yolo, yolov5, yolov8, yolov10, anchor-free, nms-free, one-stage-detector, computer-vision, deep-learning]
math: true
description: "Two-Stage Detector의 반대편, 속도를 최우선으로 하는 One-Stage Detector를 YOLO 세 버전으로 정리했다. Anchor-Based(v5) → Anchor-Free(v8) → NMS-Free(v10)로 이어지는 진화의 흐름과 실무 배포·모델 선택 기준까지 다룬다."
---

> [객체 검출(1) — Two-Stage Detector]({% post_url 2026-09-15-object-detection-1-two-stage-detector %})에서는 정확도를 최우선으로 하는 R-CNN → Fast R-CNN → Faster R-CNN의 발전 과정을 다뤘습니다. 이번 2편은 그 반대편, **속도를 최우선으로 하는 One-Stage Detector**를 다룹니다. 대표 모델인 **YOLO**의 세 핵심 버전(v5, v8, v10)이 어떻게 "느리고 정확한" Two-Stage의 구조를 참고하면서도, "빠르고 똑똑한" 방향으로 계속 진화했는지 정리합니다.
>
> IoU·NMS·mAP 같은 평가 지표는 1편에서 이미 정리했으므로, 이 글에서는 새로 등장하는 개념(CIoU, Focal Loss, Task-Aligned Assigner 등) 위주로 다룹니다.

## 📌 목차

1. [복습: Two-Stage vs One-Stage](#recap)
2. [YOLOv5: 실전 엔지니어링의 완성](#yolov5)
3. [YOLOv8: Anchor-Free 구조 혁신](#yolov8)
4. [YOLOv10: NMS-Free & End-to-End](#yolov10)
5. [Confidence Threshold와 도메인별 전략](#threshold)
6. [실무 배포: PyTorch → ONNX → TensorRT](#deploy)
7. [v5 vs v8 vs v10 한눈에 비교](#summary-table)
8. [산업별 선택 가이드](#industry-guide)
9. [핵심 요약](#summary)

---

## 1. 복습: Two-Stage vs One-Stage {#recap}

1편에서 다룬 Two-Stage Detector는 **후보 영역을 먼저 제안하고(Region Proposal), 그 다음 분류·회귀를 수행**하는 순차적 2단계 구조였습니다. One-Stage Detector는 이 두 단계를 하나의 신경망이 한 번에 처리합니다.

| 구분 | Two-Stage (R-CNN 계열) | One-Stage (YOLO, SSD) |
|---|---|---|
| 처리 방식 | 후보 제안 → 분류·회귀 (순차) | 단일 신경망이 위치+클래스 동시 예측 |
| 대표 모델 | R-CNN, Fast R-CNN, Faster R-CNN | YOLO, SSD |
| 강점 | 정확도 | 속도 |
| 약점 | 속도 | 상대적으로 낮은 정확도(격차는 계속 줄어듦) |

흥미로운 점은, 이 글에서 볼 YOLO의 진화 방향이 1편에서 본 R-CNN 계열의 진화 방향과 **같은 패턴**을 따른다는 것입니다. R-CNN → Fast R-CNN → Faster R-CNN이 "점점 더 많은 과정을 하나의 신경망 안으로 통합"했듯이, YOLO도 v5 → v8 → v10을 거치며 **Anchor 설계, 후처리(NMS)까지 순서대로 신경망 안으로 흡수**해 나갑니다.

---

## 2. YOLOv5: 실전 엔지니어링의 완성 {#yolov5}

YOLO는 "이미지를 한 번만 보고(You Only Look Once)" 위치와 클래스를 동시에 예측하는 One-Stage 모델입니다. v5는 그중에서도 **"산업 표준"**으로 자리잡을 만큼 실전성이 뛰어났던 버전입니다.

### 2.1 전체 구조: Backbone → Neck → Head

```
원본 이미지
   ↓
Backbone (CSPDarknet)   — 특징 추출
   ↓
Neck (PANet = FPN+PAN)  — 특징 다듬기
   ↓
Head (Yolo Layer)       — 최종 예측
   ↓
박스 위치 + 클래스 + 신뢰도
```

이 3단 구조는 One-Stage Detector의 표준 골격으로, v8과 v10에서도 이름만 바뀔 뿐 큰 틀은 유지됩니다.

### 2.2 Backbone: CSPDarknet과 CSP 구조

CSP(Cross Stage Partial)는 입력을 둘로 나누는 **Split & Merge** 아이디어입니다.

1. 입력 특징맵을 절반으로 **Split**
2. 절반 A는 Conv 연산(본선)을 통과
3. 절반 B는 그대로 우회(우회로)
4. 두 결과를 **Concat**해 다시 합침

연산량을 줄이면서도, 역전파 시 기울기(Gradient)가 여러 경로에서 중복 계산되는 비효율을 줄여 학습 효율을 높입니다.

### 2.3 Neck: PANet = FPN + PAN

Backbone은 깊이가 다른 여러 특징맵(대/중/소 크기)을 내놓는데, 신경망이 깊어질수록 **의미(Semantic) 정보는 강해지지만 작은 객체의 위치(Spatial) 정보는 옅어지는** 문제가 있습니다. PANet은 이를 양방향으로 보완합니다.

| 경로 | 방향 | 전달하는 정보 |
|---|---|---|
| **FPN** | Top-down (압축된 상위 → 세밀한 하위) | 의미(Semantic) 정보를 내림 |
| **PAN** | Bottom-up (세밀한 하위 → 압축된 상위) | 위치(Spatial) 정보를 올림 |

결과적으로 작은 객체(P3), 중간 객체(P4), 큰 객체(P5)를 담당하는 특징맵 모두가 "의미"와 "위치" 정보를 함께 갖추게 됩니다.

### 2.4 학습 전략: Auto Anchor + Mosaic Augmentation

v5는 모델 구조뿐 아니라 **데이터를 다루는 전략**까지 자동화했습니다.

| 기법 | 시점 | 방법 | 목적 |
|---|---|---|---|
| **Auto Anchor** | 학습 시작 전, 1회 | K-means 클러스터링으로 데이터셋에 맞는 앵커 크기 자동 산출 | 데이터 적합성 ↑ |
| **Mosaic Augmentation** | 학습 도중, 매 스텝 | 무작위 4장을 크롭·합성해 1장으로 재구성 | 배경 다양성 ↑, 소형 객체 검출력 ↑ |

두 기법은 서로 독립적으로 작동하며, 함께 v5를 "누구나 쉽게 좋은 성능을 낼 수 있는" 모델로 만들었습니다. Anchor를 데이터에서 자동으로 뽑아낸다는 점에서, 1편에서 본 Faster R-CNN의 **고정된 9개 anchor(3크기 × 3비율)**보다 한 걸음 더 데이터 친화적입니다.

---

## 3. YOLOv8: Anchor-Free 구조 혁신 {#yolov8}

v8의 핵심은 **"Anchor-Based → Anchor-Free"**로의 패러다임 전환입니다.

### 3.1 Anchor-Free: 격자 대신 좌표

| | Anchor-Based (v5) | Anchor-Free (v8) |
|---|---|---|
| 비유 | 지도 격자 — 가장 가까운 칸을 찾는다 | GPS 좌표 — 정확한 위치를 직접 찍는다 |
| 방식 | 미리 준비한 기준 박스 중 하나를 골라 조정 | 객체 중심점에서 상/하/좌/우까지의 거리를 직접 예측 |
| 필요한 것 | K-means 등 하이퍼파라미터 튜닝 | 튜닝 불필요 |

기성복(Anchor)을 골라 수선하는 방식에서, 애초에 치수를 재서 맞춤 제작(Anchor-Free)하는 방식으로 바뀐 셈입니다. 1편에서 다룬 Faster R-CNN의 anchor 9종 조합이 "얼마나 잘 맞는 기준 박스를 미리 준비하느냐"의 문제였다면, Anchor-Free는 그 준비 과정 자체를 없앤 접근입니다.

### 3.2 C2f 모듈: CSP의 업그레이드

v5의 CSP(C3)가 "본선/우회 2갈래"로만 나뉘었다면, v8의 **C2f**는 여러 단계의 중간 Block 결과를 모두 Concat합니다.

```
입력 → 1×1 Conv → Split
                     ├─ Block 1 ─┐
                     ├─ Block 2 ─┤
                     └─ Block N ─┘
        (Split 직후 값 + 모든 Block 출력) → Concat → 1×1 Conv → 출력
```

중간 Block들의 출력을 모두 모아 합치기 때문에, C3보다 더 풍부한 Gradient Flow를 확보합니다.

### 3.3 Decoupled Head: 판사와 측량사를 분리하다

기존(Coupled Head)은 분류(Classification)와 위치 회귀(Box Regression)가 같은 레이어를 공유했습니다. 그런데 이 둘은 사실 **서로 다른 것에 집중해야 하는 작업**입니다.

- 분류가 원하는 것: 물체의 중심부, 특징적인 부분
- 위치 회귀가 원하는 것: 물체의 정확한 경계, 가장자리

같은 레이어에 억지로 태우면 두 목표가 경쟁하며 학습이 방해받는 **Task Misalignment**가 발생합니다. v8은 두 브랜치를 물리적으로 분리한 **Decoupled Head**로 이 문제를 해결했습니다.

```
Neck 출력 특징맵
   ├─→ 분류 브랜치(Classification) → 클래스 확률
   └─→ 위치 브랜치(Box Regression) → 박스 좌표
```

### 3.4 학습 전략: Task-Aligned Assigner + CIoU/Focal Loss

**Task-Aligned Assigner**는 "어떤 grid 칸이 이 객체를 책임질지" 정할 때, 분류 점수(s)와 IoU(u)를 곱한 통합 점수로 판단합니다.

$$t = s^\alpha \times u^\beta$$

분류도 잘하고 위치도 잘 맞는 칸을 우선적으로 담당자로 선정하는 방식입니다. 이 IoU(u)는 1편 5.1절에서 다룬 것과 같은 개념으로, 이번에는 "학습 대상 grid를 정하는 기준"으로 재사용됩니다.

선정된 칸은 두 가지 손실 함수로 학습됩니다.

- **CIoU Loss (위치)**

$$Loss_{CIoU} = 1 - IoU + \frac{\rho^2(b, b_{gt})}{c^2} + \alpha v$$

겹침(IoU)뿐 아니라 중심점 거리, 종횡비까지 고려합니다. 1편에서 본 gIoU가 "멀리 떨어진 정도"를 반영했다면, CIoU는 여기에 중심점 거리와 종횡비 페널티까지 더한 발전형입니다.

- **Focal Loss (분류)**

$$FL = -\alpha(1-p)^\gamma \log(p)$$

쉬운 배경 샘플의 손실 기여도를 줄이고 어려운 객체 샘플에 학습을 집중시킵니다. 1편 9절에서 본 Faster R-CNN의 "positive/negative 256개 샘플링"이 배경 편중 문제를 **샘플 개수 조절**로 풀었다면, Focal Loss는 **손실 함수 자체의 가중치 조절**로 같은 문제를 풉니다.

---

## 4. YOLOv10: NMS-Free & End-to-End {#yolov10}

v10은 검출 파이프라인의 마지막 남은 후처리 단계, **NMS**를 학습 과정에 내재화해 완전히 제거했습니다.

### 4.1 기존 NMS의 한계

| 한계 | 내용 |
|---|---|
| 민감한 하이퍼파라미터 | IoU Threshold 설정에 따라 결과가 크게 달라짐 |
| 추론 지연 시간 | GPU 병렬 연산 후, NMS는 순차적으로 처리해야 해 병목 발생 |
| 배포 복잡성 | TensorRT 등 최적화 시 NMS 플러그인 구현이 까다로움 |

이 한계는 1편에서 본 Two-Stage의 병목과 정확히 대응됩니다. Faster R-CNN이 "CPU에서 도는 Selective Search"를 병목으로 지목하고 RPN으로 대체했듯, v10은 "순차 처리되는 NMS"를 병목으로 지목하고 아예 제거합니다.

### 4.2 Dual Label Assignment: 예선과 본선

v10은 헤드를 두 개 두어 학습과 추론의 역할을 분리합니다.

| 헤드 | 역할 | 특징 |
|---|---|---|
| **One-to-Many** (예선) | 학습에만 관여 | 하나의 정답에 여러 박스 할당 → 학습 신호 풍부(Recall↑) |
| **One-to-One** (본선) | 추론 시 이것만 사용 | 정답 하나에 박스 1개만 매칭 → 중복이 애초에 생기지 않음 |

```
기존(v5/v8): 예측 → NMS → 결과
v10(NMS-Free): 예측 → 결과 (즉시)
```

### 4.3 아키텍처 효율화: Large-kernel DWConv + CIB + PSA

v10은 후처리뿐 아니라 **모델 내부 연산 자체**도 가볍게 만들었습니다.

**왜 다시 큰 커널(7×7)을 쓸까?**

일반 Convolution에서는 큰 커널일수록 파라미터가 `커널크기 × 커널크기 × 입력채널 × 출력채널`로 폭증해 무겁습니다. 그래서 기존 CNN들은 3×3을 여러 번 쌓는 방식을 선호했습니다. 하지만 **Depthwise Convolution**은 채널별로 독립 계산해 파라미터가 `커널크기 × 커널크기 × 채널수`로 훨씬 적습니다. 따라서 큰 커널을 Depthwise로 적용하면, 넓은 Receptive Field(수용 영역)를 적은 레이어로 확보하면서도 연산은 가볍게 유지할 수 있습니다.

**CIB (Compact Inverted Block)** 구조:

```
입력 → 1×1 Conv(확장, Expand) → Depthwise Conv(심화) → 1×1 Conv(축소, Project) → 출력
```

채널을 넓혀 여유 있는 표현 공간을 확보한 뒤(확장), 그 공간에서 가볍게 특징을 추출하고(Depthwise), 다시 필요한 크기로 압축(축소)하는 구조로, 경량화와 특징 보존의 균형을 잡습니다.

**PSA (Partial Self-Attention)**: Self-Attention은 이미지 전역의 관계(어느 영역이 다른 어느 영역과 관련 있는지)를 포착하는 데 강력하지만, 모든 채널에 적용하면 연산량이 급격히 늘어납니다. PSA는 CSP처럼 채널을 Split해 **절반만 Self-Attention에 통과**시키고 나머지 절반은 그대로 우회시킨 뒤 Concat합니다. CIB가 지역적(local) 특징을 가볍게 뽑는 역할이라면, PSA는 전역적(global) 문맥 정보를 적은 비용으로 보완하는 역할을 맡습니다.

---

## 5. Confidence Threshold와 도메인별 전략 {#threshold}

1편에서 Precision·Recall·mAP의 정의와 계산법을 다뤘으니, 여기서는 실무에서 **Confidence Threshold를 어디에 두느냐**가 왜 도메인마다 다른 선택이 되는지에 집중합니다.

| Threshold 조정 | Recall | Precision |
|---|---|---|
| 낮춤 | ↑ (놓치는 것↓) | ↓ (오탐↑) |
| 높임 | ↓ (미탐↑) | ↑ (정확한 것만) |

| 상황 | 전략 | 이유 |
|---|---|---|
| 의료 영상 진단 | Recall 우선 (Threshold 낮게) | 놓치면(미탐) 생명 위협 |
| 보안 CCTV | Precision 우선 (Threshold 높게) | 오탐 시 불필요한 출동 비용 발생 |
| 일반 검색/분류 | 균형(Balanced) | F1-Score 최대화 지점 사용 |

같은 모델이라도 threshold 하나로 완전히 다른 제품이 될 수 있다는 점이 핵심입니다.

---

## 6. 실무 배포: PyTorch → ONNX → TensorRT {#deploy}

학습이 끝난 모델을 실제 서비스에 배포할 때는, 모델 형식(엔진)을 바꿔가며 추론 속도를 추가로 최적화합니다.

```
PyTorch (.pt)   ~150ms
   ↓ ONNX 변환
ONNX            ~75ms   (2배)
   ↓ TensorRT 최적화
TensorRT        ~20ms   (7.5배)
```

- **PyTorch(.pt)**: 학습·실험에 유리한 기본 형식
- **ONNX**: 여러 프레임워크에서 공통으로 쓸 수 있는 범용 중간 표현
- **TensorRT**: NVIDIA GPU에 특화된 고속 추론 엔진 (레이어 결합, 정밀도 최적화 등 수행)

> 비유하자면 PyTorch는 다양한 길을 갈 수 있는 승용차, TensorRT는 특정 서킷(GPU)에 맞춰 불필요한 부품을 모두 떼어낸 F1 머신입니다.

---

## 7. v5 vs v8 vs v10 한눈에 비교 {#summary-table}

| 항목 | YOLOv5 | YOLOv8 | YOLOv10 |
|---|---|---|---|
| 아키텍처 | CSP + PANet(FPN+PAN) | C2f + PANet(FPN+PAN) | CIB + PSA(Partial Self-Attention) |
| 앵커 방식 | Anchor-Based | Anchor-Free | Anchor-Free |
| 헤드 구조 | Coupled Head | Decoupled Head | Decoupled Head (Dual: One-to-Many + One-to-One) |
| 후처리 | NMS 필수 | NMS 필수 | **NMS-Free** |
| 손실함수 | GIoU/CIoU | CIoU + Focal + DFL | CIoU + Focal + DFL (동일, Assignment만 이원화) |
| 추천 용도 | 안정성, 레거시 호환 | 고성능 표준, 쉬운 학습 | 초저지연, 엣지 디바이스 |

```
v5   Anchor-Based    (실전 엔지니어링)
  ↓
v8   Anchor-Free     (구조적 유연함)
  ↓
v10  NMS-Free        (효율적 종단학습)
```

1편의 R-CNN 계열 표와 나란히 놓고 보면 궤적은 같지만(1장 참고), 남은 차이는 **아직 흡수되지 않은 단계가 있는지**입니다. R-CNN 계열은 Faster R-CNN에서 Region Proposal(RPN)까지 흡수하며 사실상 완전한 end-to-end에 도달했지만, YOLO 계열은 v10에서 NMS는 제거했어도 Anchor-Free 전환(v8)과 NMS 제거(v10)가 서로 다른 버전에 걸쳐 이뤄졌다는 점이 다릅니다.

---

## 8. 산업별 선택 가이드 {#industry-guide}

모델 자체의 특성만 비교하며, threshold를 어느 쪽으로 둘지는 5장 기준을 그대로 적용하면 됩니다.

| 산업 | 추천 모델 | 이유 |
|---|---|---|
| 자율주행 | v8 / v10 (L/M) | 밀집 객체 대응(Anchor-Free, NMS-Free), 지연시간 최소화 |
| 의료 영상 | v8 (L/X) | Focal Loss로 데이터 불균형(희귀 질환) 학습에 효과적 |
| 보안 CCTV | v5 (M) / v8 (S) | 검증된 파이프라인, 24시간 가동 대비 가성비 |
| 모바일·엣지 | v5n / v8n / v10n | 리소스 제약 대응, v10n은 CIB로 연산 대비 높은 mAP |

절대적으로 가장 좋은 모델은 없습니다. **현장의 제약(하드웨어, 지연시간, 데이터 특성)에 맞춰 v5·v8·v10 중 무엇을 선택할지가 실무 성능의 절반을 결정**합니다.

---

## 9. 핵심 요약 {#summary}

**One-Stage 철학**
Region Proposal과 Classification을 분리하지 않고 단일 신경망이 한 번에 처리하여 속도를 최우선으로 추구합니다. 실시간성이 중요한 자율주행, CCTV, 엣지 디바이스에 적합합니다.

**YOLO의 진화**
v5(Anchor-Based) → v8(Anchor-Free) → v10(NMS-Free)로 발전하며 튜닝 포인트와 후처리 비용을 순서대로 줄여왔지만, 그만큼 구조는 복잡해졌습니다. 안정성이 최우선이라면 여전히 v5도 유효한 선택입니다.

**핵심 체크리스트**

- [ ] One-Stage Detector는 후보 제안과 분류·회귀를 하나의 신경망이 한 번에 처리해 속도를 우선한다
- [ ] YOLOv5는 CSPDarknet(Backbone) + PANet(Neck) + Yolo Layer(Head) 구조이며, Auto Anchor와 Mosaic Augmentation으로 학습을 자동화했다
- [ ] CSP는 입력을 Split해 절반은 Conv, 절반은 우회시킨 뒤 Concat해 연산량과 Gradient 중복을 줄인다
- [ ] PANet은 FPN(Top-down, 의미 전달)과 PAN(Bottom-up, 위치 전달)을 결합해 모든 크기의 객체에 균형 잡힌 특징을 제공한다
- [ ] YOLOv8은 Anchor-Free로 전환해 K-means 튜닝 없이 중심점에서 거리를 직접 예측한다
- [ ] Decoupled Head는 분류와 위치 회귀를 물리적으로 분리해 Task Misalignment 문제를 해결한다
- [ ] Task-Aligned Assigner는 분류 점수와 IoU를 곱한 통합 점수로 학습 담당 grid를 정한다
- [ ] Focal Loss는 쉬운 배경 샘플의 기여도를 줄이고 어려운 샘플에 학습을 집중시켜 클래스 불균형을 완화한다
- [ ] YOLOv10은 One-to-Many(학습용)와 One-to-One(추론용) 두 헤드로 NMS 없이 중복 없는 예측을 만든다
- [ ] YOLOv10은 CIB(지역 특징, Depthwise 기반)와 PSA(전역 문맥, 채널 절반만 Self-Attention)를 함께 써서 연산 대비 표현력을 확보한다
- [ ] Confidence Threshold는 Precision-Recall 트레이드오프를 조절하는 손잡이이며, 의료(Recall 우선)와 보안(Precision 우선)처럼 도메인마다 최적점이 다르다
- [ ] 실무 배포에서는 PyTorch → ONNX → TensorRT 순으로 변환하며 추론 속도를 최적화한다

---

*이 글은 YOLO v5, v8, v10의 구조 변화(Anchor-Based → Anchor-Free → NMS-Free)와 실무 배포·모델 선택 기준을 정리한 학습 노트입니다. [1편(Two-Stage Detector)]({% post_url 2026-09-15-object-detection-1-two-stage-detector %})과 함께 보면 R-CNN 계열과 YOLO 계열이 각자 다른 출발점에서 결국 비슷한 방향(신경망 내부로의 통합)으로 수렴해온 과정을 비교할 수 있습니다.*
