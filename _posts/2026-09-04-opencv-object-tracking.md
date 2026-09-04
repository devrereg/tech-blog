---
title: "OpenCV 스터디 4주차: 템플릿 매칭부터 RAFT까지, 객체 추적 알고리즘 총정리"
date: 2026-09-04 09:30:00 +0900
categories: [AI, Computer Vision]
tags: [opencv, computer-vision, object-tracking, template-matching, feature-matching, meanshift, camshift, optical-flow, lucas-kanade, dense-optical-flow, raft, deep-learning]
math: true
description: "영상 속 움직이는 객체를 추적하는 기술을, 가장 단순한 템플릿 매칭에서 출발해 특징점 매칭, MeanShift/CamShift, Lucas-Kanade Optical Flow, Dense Optical Flow, 그리고 딥러닝 기반 RAFT까지 하나의 흐름으로 정리했다. '픽셀을 그대로 비교하는 방식 → 고유한 지점을 비교하는 방식 → 분포의 무게중심을 쫓는 방식 → 점의 이동을 수식으로 추적하는 방식 → 이 모든 과정을 데이터로 학습한 방식'이라는 발전 관계를 실습 코드와 함께 짚는다."
---

> 이 글은 OpenCV 스터디 4주차 "객체 추적(Object Tracking)" 내용을 정리한 노트입니다. 가장 기초적인 **템플릿 매칭**부터 시작해서 **특징점 매칭**, **MeanShift/CamShift**, **Lucas-Kanade Optical Flow**, **Dense Optical Flow**, 그리고 최신 **딥러닝 기반 RAFT**까지, 추적 알고리즘이 어떤 순서로 발전해 왔는지 하나의 흐름으로 정리합니다.
>
> [1주차 정리]({% post_url 2026-09-01-opencv-digital-image-basics %})가 "영상이 어떻게 픽셀 행렬이 되는가", [2주차 정리]({% post_url 2026-09-01-opencv-geometric-transformation %})가 "그 행렬의 값과 좌표를 바꾸는" 이야기, [3주차 정리]({% post_url 2026-09-02-opencv-feature-detection %})가 "행렬 위에서 의미 있는 지점(엣지·코너)을 찾는" 이야기였다면, 이번 글은 그 지점들을 **시간 축을 따라 계속 쫓아가는 방법** — 즉 추적(Tracking) — 을 다룹니다.

## 📌 목차

1. [영상 속 객체를 추적한다는 것](#intro)
2. [템플릿 매칭: 가장 기초적인 방법](#template-matching)
3. [특징점 추출과 매칭: 픽셀 대신 "고유한 지점"을 비교하자](#feature-matching)
4. [MeanShift: 밀도 분포의 무게중심을 쫓아가기](#meanshift)
5. [CamShift: 크기와 회전을 적응시키는 MeanShift](#camshift)
6. [Lucas-Kanade Optical Flow: 점의 움직임을 수식으로 추적하기](#lucas-kanade)
7. [Dense Optical Flow: 특징점이 아닌 모든 픽셀을 추적하기](#dense-optical-flow)
8. [RAFT: 딥러닝으로 진화한 Optical Flow](#raft)
9. [전체 기법 비교 및 선택 가이드](#comparison)
10. [마무리](#summary)

---

## 1. 영상 속 객체를 추적한다는 것 {#intro}

영상에서 움직이는 객체를 추적하는 기술은 자율주행, CCTV 감시, 스포츠 분석, 어린이 보호 등 다양한 곳에 쓰입니다. 이 글에서는 가장 기초적인 **템플릿 매칭**부터 시작해서, **고유한 특징점 기반 매칭**, **밀도 분포 기반 추적(MeanShift/CamShift)**, **특징점 기반 Optical Flow(Lucas-Kanade)**, 그리고 최신 **딥러닝 기반 Optical Flow(RAFT)**까지, 추적 알고리즘의 발전 흐름을 순서대로 정리합니다.

![템플릿 매칭 → 특징점 검출·매칭 → MeanShift/CamShift → Lucas-Kanade(Sparse Flow) → Dense Optical Flow → RAFT(딥러닝)로 이어지는 6단계 추적 알고리즘 로드맵을 화살표로 연결한 다이어그램. 하단에는 픽셀 기반 → 특징 기반 → 분포 기반 → 모션 벡터 기반 → 딥러닝 기반이라는 발전 축이 표시되어 있다](/assets/img/posts/opencv-object-tracking/01_roadmap.svg)

*추적 알고리즘 전체 로드맵 — "무엇을 기준으로, 어떻게 비교할 것인가"가 계속 달라지며 발전한다*

전체 흐름을 한 문장으로 요약하면 다음과 같습니다.

> **픽셀을 그대로 비교하는 방식(템플릿 매칭) → 고유한 지점을 비교하는 방식(특징점 매칭) → 밀도 분포의 무게중심을 쫓는 방식(MeanShift/CamShift) → 점의 이동을 수식으로 추적하는 방식(Optical Flow) → 이 모든 과정을 데이터로 학습한 방식(RAFT)**

---

## 2. 템플릿 매칭: 가장 기초적인 방법 {#template-matching}

**템플릿 매칭(Template Matching)**은 **작은 조각 이미지(템플릿)**를 원본 이미지 위에서 한 칸씩 이동시키며, 가장 비슷한 위치를 찾는 기법입니다. 자리를 옮겨가며 스캔한다는 점에서 슬라이딩 윈도우 방식이라고 볼 수 있습니다. 템플릿을 원본 위에 슬라이딩하면서 매 위치마다 유사도 점수를 계산하고, 그 점수들을 모은 결과 행렬에서 가장 튀는 지점(최댓값 또는 최솟값)을 찾으면 그것이 정답 위치입니다.

```python
res = cv2.matchTemplate(image, template, method)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
```

`method`에 따라 결과를 해석하는 방식이 달라집니다. 상관계수 기반(`TM_CCOEFF`, `TM_CCORR` 등)은 값이 클수록 유사하고, 오차 기반(`TM_SQDIFF`)은 값이 작을수록 유사합니다. 실무에서는 밝기 변화에 강하고 0~1 범위로 해석이 쉬운 `TM_CCOEFF_NORMED`가 가장 널리 쓰입니다.

**한계**: 템플릿 매칭은 이미지 전체 영역을 픽셀 단위로 그대로 비교하기 때문에, 조명이 바뀌거나 객체가 회전·확대/축소되면 픽셀 배열 자체가 달라져서 매칭에 실패합니다.

---

## 3. 특징점 추출과 매칭: 픽셀 대신 "고유한 지점"을 비교하자 {#feature-matching}

템플릿 매칭의 한계를 극복하기 위해, 이미지 전체가 아니라 **강하게 구별되는 몇몇 지점(특징점, Keypoint)**만 뽑아서 비교하는 방식이 등장했습니다.

### 왜 하필 "모서리(Corner)"가 좋은 특징점일까?

![평탄한 영역, 경계선, 모서리 세 가지 경우에서 중심점을 기준으로 여러 방향의 화살표를 그려 밝기 변화를 비교하는 다이어그램. 평탄한 영역은 모든 방향에서 변화가 같아 회색 화살표로 표시되고, 경계선은 경계를 따라서는 변화가 없어 위아래로만 초록 화살표가 표시되며, 모서리는 모든 방향에서 뚜렷하게 달라져 여덟 방향 모두 초록 화살표로 표시된다](/assets/img/posts/opencv-object-tracking/02_corner_concept.svg)

*평탄한 영역은 위치를 특정할 수 없고, 경계선은 방향에 따라 모호하며(구멍 문제), 모서리만이 모든 방향에서 뚜렷하게 구별된다*

- **평탄한 영역(Flat)**: 어느 방향으로 봐도 주변과 구별이 안 됨 → 특징점으로 부적합
- **경계선(Edge)**: 선을 따라서는 구별이 안 됨(구멍 문제, Aperture Problem) → 애매함
- **모서리(Corner)**: 어느 방향으로 이동해도 확실히 구별됨 → 가장 좋은 특징점

### 특징점 매칭의 처리 과정

```
입력 영상 → 특징점 검출(Keypoint) → 디스크립터화(Descriptor) → 매칭(Matching)
```

1. **특징점 검출**: 모서리 같은 강한 지점의 위치를 찾음
2. **디스크립터 생성**: 그 지점 주변의 패턴을 숫자 벡터로 표현
3. **매칭**: 두 이미지의 디스크립터 벡터끼리 유사도를 비교해 짝을 찾음

이 방식이 강인한 이유는, 픽셀 값 자체가 아니라 **그 지점 주변의 상대적인 패턴**을 벡터로 표현하기 때문입니다. 조명이 바뀌거나 이미지가 회전해도 "여기는 이러한 배경을 낀 모서리가 있다"는 상대적 패턴은 유지되므로 매칭이 안정적일 수 있습니다.

### 대표적인 디스크립터 알고리즘

| 시기 | 대표 알고리즘 | 핵심 아이디어 |
|---|---|---|
| 1999~2009 (정확도 중심) | SIFT, SURF, DAISY | 그래디언트 방향·크기를 히스토그램으로 표현 |
| 2010~2014 (속도 중심) | BRIEF, ORB, FREAK, AKAZE | 이진 코드 비교로 가볍고 빠르게 처리 |
| 2015~ (딥러닝 시대) | LATCH, LoFTR(Transformer) | 패치 패턴을 직접 학습 |

실무에서 가장 널리 쓰이는 건 **ORB**입니다. 특허 문제 없이 무료로 쓸 수 있고, 회전·스케일 변화에도 강해서 OpenCV의 기본 선택지로 인기가 많습니다.

---

## 4. MeanShift: 밀도 분포의 무게중심을 쫓아가기 {#meanshift}

### 핵심 아이디어

> **데이터가 밀집된 곳(무게중심, Mode)으로 탐색 윈도우를 반복적으로 이동시켜 객체를 찾는다.**

"Mean(평균/무게중심)"으로 "Shift(이동)"한다는 이름 그대로, 색이 밀집된 방향으로 박스를 계속 옮겨가는 알고리즘입니다.

### 동작 4단계

1. **색상 모델 생성**: 추적할 객체 영역(ROI)의 색상 히스토그램을 계산
2. **유사 영역 투표(Back Projection)**: 전체 영상에서 이 히스토그램과 비슷한 색 분포를 가진 곳을 확률 맵으로 표현
3. **중심 이동(Shift)**: 현재 윈도우 내 픽셀들의 무게중심을 계산해서 그 방향으로 윈도우를 이동
4. **반복 수행**: 중심 이동 거리가 거의 0에 수렴할 때까지 3단계를 반복

### MeanShift의 한계

MeanShift는 탐색 윈도우의 **크기와 방향이 고정**되어 있습니다. 그래서 객체가 카메라에 가까워지며 커지거나 회전하면, 박스가 객체를 제대로 감싸지 못하는 문제가 생깁니다.

---

## 5. CamShift: 크기와 회전을 적응시키는 MeanShift {#camshift}

**CamShift**는 **C**ontinuously **A**daptive **M**ean**Shift**의 약자로, 이름 그대로 MeanShift에 "적응형 윈도우" 기능을 추가한 개선 버전입니다.

![상단 행은 고정 크기의 빨간 추적 박스가 객체가 커지는 3개 프레임 동안 크기를 바꾸지 못해 점점 객체를 벗어나는 모습을, 하단 행은 초록 추적 박스가 객체 크기에 맞춰 함께 커지며 계속 감싸는 모습을 보여주는 비교 다이어그램](/assets/img/posts/opencv-object-tracking/03_meanshift_camshift.svg)

*MeanShift는 윈도우 크기가 고정돼 있어 객체가 커지면 놓치지만, CamShift는 2차 모멘트로 크기와 방향을 매 프레임 다시 계산한다*

MeanShift의 흐름을 거의 갖지만, 딱 한 단계가 추가됩니다.

| 단계 | MeanShift | CamShift |
|---|---|---|
| 1 | 색상 모델 생성 | 색상 모델 생성 (HSV의 Hue 히스토그램 사용) |
| 2 | Back Projection | Back Projection (동일) |
| 3 | 중심 이동 | MeanShift 수행 (동일) |
| 4 | *(없음)* | **윈도우 크기·방향 업데이트 (2차 모멘트 계산)** |
| 5 | 반복 | 반복 |

4단계에서 윈도우 안의 픽셀 분포를 통계적으로 분석(2차 모멘트)해서, 그 분포가 넓게 퍼져 있으면 윈도우를 키우고, 좁으면 줄이고, 기울어져 있으면 각도까지 조절합니다. 그 결과 **객체가 이동·확대·축소·회전해도 안정적으로 추적**할 수 있습니다.

> **참고**: RGB 대신 **HSV 색공간의 Hue(색상)** 채널을 주로 사용하는 이유는, Hue가 밝기(Brightness) 정보와 분리되어 있어서 조명 변화에도 상대적으로 안정적이기 때문입니다.

---

## 6. Lucas-Kanade Optical Flow: 점의 움직임을 수식으로 추적하기 {#lucas-kanade}

### MeanShift와는 근본적으로 다른 접근

MeanShift/CamShift가 "색상 분포"를 쫓는다면, Optical Flow는 **개별 특징점(코너 등)이 다음 프레임에서 어디로 이동했는가**를 수식으로 계산합니다.

### 핵심 수식

$$I(x, y, t) = I(x+dx, y+dy, t+dt)$$

시간 $t$에서 $(x, y)$ 위치의 밝기값이, 아주 짧은 시간 뒤($t+dt$) $(x+dx, y+dy)$ 위치의 밝기값과 같다는 뜻입니다. 우리가 구해야 할 미지수는 이 점이 얼마나 이동했는지를 나타내는 **dx, dy**입니다.

### 이 문제를 풀기 위한 3가지 가정

이 식 하나만으로는 미지수 2개(dx, dy)를 구할 수 없기 때문에, 3가지 표준적인 가정을 추가로 도입합니다.

![밝기 보존 가정을 두 개의 어두운 사각형과 화살표로, 미소 이동 가정을 짧은 시간 간격의 타임라인과 작은 원 이동으로, 공간적 일관성 가정을 함께 같은 방향으로 움직이는 3x3 초록 화살표 격자로 표현한 3단 다이어그램](/assets/img/posts/opencv-object-tracking/04_lk_assumptions.svg)

*Lucas-Kanade는 밝기 보존, 미소 이동, 공간적 일관성이라는 세 가정 위에서 방정식을 풀 수 있게 만든다*

1. **밝기 보존 가정**: 물체의 한 점의 시간이 지나도 밝기값이 변하지 않는다 → 식의 성립 그 자체
2. **미소 이동 가정**: 프레임 사이의 이동량이 매우 작다(Δx ≈ 0) → 테일러 급수로 식을 선형 근사할 수 있게 해줌
3. **공간적 일관성 가정**: 이웃 픽셀들은 비슷한 움직임을 가진다 → 이웃 픽셀 수만큼 방정식을 추가로 만들어서, 미지수보다 방정식이 많아지게 함(최소제곱법으로 풀이 가능해짐)

세 가정은 각각 다른 역할을 하며, "이론적으로 풀기 어려운 문제"를 "실제로 계산 가능한 문제"로 바꿔줍니다.

### 가정을 적용해 실제 방정식 세우기

밝기 보존 가정과 미소 이동 가정을 합치면, 테일러 급수로 1차 항까지만 남기고 근사할 수 있습니다.

$$I(x+dx, y+dy, t+dt) \approx I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt$$

밝기 보존 가정에 의해 좌변($I(x+dx, y+dy, t+dt)$)과 $I(x,y,t)$가 같으므로, 나머지 항의 합이 0이 됩니다.

$$I_x \, dx + I_y \, dy + I_t \, dt = 0$$

양변을 $dt$로 나누고, 픽셀의 이동 속도를 $u = dx/dt$, $v = dy/dt$로 정의하면 다음과 같은 **광학 흐름 제약 방정식(Optical Flow Constraint Equation)**이 나옵니다.

$$I_x u + I_y v + I_t = 0$$

여기서 $I_x, I_y$는 각각 x, y 방향 그래디언트(Sobel로 계산 가능), $I_t$는 프레임 간 시간 차분입니다. 그런데 이 식은 **방정식 1개에 미지수가 $u, v$ 2개** — 3절에서 다룬 구멍 문제(Aperture Problem)가 여기서 그대로 다시 등장합니다. 그래디언트 하나만으로는 그 방향을 따라 얼마나 움직였는지 알 수 없습니다.

이때 세 번째 가정인 **공간적 일관성**이 힘을 발휘합니다. 윈도우 안의 이웃 픽셀 $n$개가 모두 같은 $(u, v)$로 움직인다고 가정하면, 같은 형태의 방정식을 $n$개 얻을 수 있습니다.

$$\begin{bmatrix} I_{x1} & I_{y1} \\ I_{x2} & I_{y2} \\ \vdots & \vdots \\ I_{xn} & I_{yn} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t1} \\ I_{t2} \\ \vdots \\ I_{tn} \end{bmatrix}$$

미지수(2개)보다 방정식($n$개)이 더 많은 **과결정계(Overdetermined System)**가 되므로, 최소제곱법으로 오차를 최소화하는 $(u, v)$를 구할 수 있습니다. 양변에 계수 행렬의 전치를 곱해 정리하면, 최종적으로 풀어야 할 식은 다음과 같습니다.

$$\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}$$

> **3절과의 연결고리**: 왼쪽의 2×2 행렬은 Harris 코너 검출에서 쓰이는 구조 텐서(Structure Tensor)와 같은 형태입니다. 이 행렬이 역행렬을 가지려면(두 방향 모두 고유값이 충분히 커야 함) 그래디언트가 여러 방향으로 골고루 강해야 하는데, 이는 정확히 "모서리(Corner)"의 조건입니다. Lucas-Kanade가 평탄한 영역이나 경계선이 아니라 굳이 코너를 추적점으로 고르는 이유가 바로 여기서 나옵니다.

---

## 7. Dense Optical Flow: 특징점이 아닌 모든 픽셀을 추적하기 {#dense-optical-flow}

Lucas-Kanade가 **Sparse(희소)**하게 몇 개의 특징점만 추적한다면, **Dense Optical Flow**는 영상 안의 **모든 픽셀**에 대해 움직임 벡터를 계산합니다.

![왼쪽은 몇 개의 빨간 점과 짧은 화살표만으로 표현된 Sparse Optical Flow, 오른쪽은 격자 전체를 채운 다양한 색상의 작은 화살표들로 표현된 Dense Optical Flow를 나란히 비교하는 다이어그램](/assets/img/posts/opencv-object-tracking/05_sparse_vs_dense.svg)

*Sparse는 몇 개의 특징점만, Dense는 모든 픽셀의 움직임 벡터를 계산한다*

각 픽셀 (x, y)마다 x축 속도(u)와 y축 속도(v)를 포함한 2차원 벡터를 계산하며, 이 결과를 사람이 한눈에 보기 좋게 **HSV 색상 맵**으로 표현합니다.

| HSV 성분 | 의미 |
|---|---|
| Hue(색상) | 움직임의 방향 |
| Value(명도) | 움직임의 속도 |

배경(하늘, 벽, 도로 등 특징점이 없는 영역)의 움직임까지 파악할 수 있어서, 동작 인식이나 영상 분할(Segmentation), 자율주행처럼 **장면 전체를 이해해야 하는 작업**에 활용됩니다. 다만 모든 픽셀을 계산하기 때문에 Sparse 방식보다 훨씬 무겁습니다.

---

## 8. RAFT: 딥러닝으로 진화한 Optical Flow {#raft}

### RAFT란?

**RAFT(Recurrent All-Pairs Field Transforms)**는 2020년 프린스턴 대학이 발표했고, ECCV 2020 Best Paper로 선정된 딥러닝 기반 Dense Optical Flow 모델입니다. 발표 당시 KITTI, Sintel 등 표준 벤치마크에서 기존 최고 기록 대비 16~30%나 오차를 줄이며 압도적인 성능을 보여줬습니다.

### 왜 필요했을까?

기존 Optical Flow는 밝기 보존, 미소 이동, 공간적 일관성 같은 **사람이 설계한 가정**에 의존했습니다. 이 가정들은 카메라가 빠르게 움직이거나, 물체가 가려지거나(Occlusion), 무늬 없는 표면처럼 예외적인 상황에서 쉽게 깨졌습니다. RAFT는 이 가정들을 사람의 손으로 설계하는 대신, **대량의 데이터로 학습한 신경망**이 이런 예외 상황까지 처리하도록 만들었습니다.

### 아키텍처 4단계

![Frame 1과 Frame 2가 Feature Encoder(CNN)로 들어가 4D Correlation Volume을 거치고, GRU Update Block이 Context Encoder의 정보를 받아 12회 반복 정제한 뒤 Optical Flow(dx, dy) 결과를 출력하는 아키텍처 흐름도](/assets/img/posts/opencv-object-tracking/06_raft_architecture.svg)

*모든 픽셀 쌍의 상관관계를 4D 볼륨에 담고, GRU가 흐름장을 0에서부터 반복적으로 정제한다*

1. **Feature Encoder(CNN)**: 두 프레임 각각의 **모든 픽셀**에서 픽셀 값이 아닌 의미 있는 특징(feature) 벡터를 추출
2. **4D Correlation Volume**: Frame 1의 **모든** 픽셀과 Frame 2의 **모든** 픽셀 사이의 유사도를 전부 계산(다중 스케일로 계산해서 크고 작은 움직임을 동시에 커버)
3. **Context Encoder**: Frame 1의 문맥 정보를 별도로 추출해서, 반복 정제 단계에서 계속 참조할 수 있게 제공
4. **GRU Update Block**: 흐름 필드를 **0(움직임 없음)**에서 시작해, GRU(순환 신경망)가 상관관계 정보를 조회하며 약 12회 반복적으로 정제

> 마치 사람이 스케치를 먼저 그리고 세부 묘사를 점점 다듬어가는 과정과 비슷합니다. 처음엔 큰 흐름만 대략 잡고, 반복할 수록 점점 정교해집니다.

### 코드로 보면 이렇게 동작합니다

```python
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()

with torch.no_grad():
    list_of_flows = model(img1_tensor, img2_tensor)

predicted_flow = list_of_flows[-1]   # 12번의 반복 중 마지막(가장 정제된) 결과
# shape: [배치, 2, H, W] → 채널 2는 각각 dx, dy
```

`list_of_flows`가 리스트인 이유가 바로 GRU의 반복 정제 구조 때문입니다. 각 원소가 한 번의 반복(iteration) 결과이고, 마지막 원소가 가장 정교하게 다듬어진 최종 흐름입니다.

---

## 9. 전체 기법 비교 및 선택 가이드 {#comparison}

| 기법 | 비교 기준 | 장점 | 단점 | 적합한 상황 |
|---|---|---|---|---|
| **템플릿 매칭** | 픽셀 전체 | 간단하고 직관적 | 크기·회전·밝기 변화에 매우 취약 | 고정된 아이콘/로고 검출 |
| **MeanShift** | 색상 히스토그램 | 빠른 속도 | 크기 변화·회전을 못 잡음 | 단순 소형 객체 추적 |
| **CamShift** | 색상 히스토그램 | 크기·회전을 유연하게 적응 | 배경과 색상이 비슷하면 실패 | 거리 변화가 있는 얼굴/객체 추적 |
| **Lucas-Kanade** | 특징점(Sparse) | 정밀하고 실시간 처리 가능 | 큰 움직임(Large Motion)에 취약 | 영상 안정화, 특징점 추적 |
| **Dense Optical Flow** | 모든 픽셀 | 배경 포함 전체 움직임 파악 | 계산 비용이 매우 큼 | 동작 인식, 영상 분할 |
| **RAFT** | 모든 픽셀(딥러닝) | 현존 최고 수준의 정확도·강건함 | 높은 GPU 성능 필요 | 자율주행 연구, 고정밀 비전 시스템 |

### 발전 관계로 요약하면

```
템플릿 매칭 ──(속도 개선)───▶ MeanShift ──(유연성 개선)───▶ CamShift
Lucas-Kanade ──(커버리지 확장)───▶ Dense Optical Flow ──(정확도 개선, 딥러닝)───▶ RAFT
```

---

## 10. 마무리 {#summary}

이번 글에서 다룬 내용을 한 문장씩 정리하면 다음과 같습니다.

- **템플릿 매칭**: 조각 이미지를 슬라이딩하며 픽셀 단위로 비교하는 가장 기초적인 방법
- **특징점 매칭**: 모서리 같은 강한 지점만 뽑아 디스크립터로 변환한 뒤 비교, ORB가 실무에서 가장 대중적
- **MeanShift**: 색상 분포의 무게중심을 반복적으로 쫓아가지만, 윈도우 크기·방향이 고정되어 있음
- **CamShift**: MeanShift에 2차 모멘트 기반 윈도우 적응 단계를 추가해 크기·회전 변화에 대응
- **Lucas-Kanade Optical Flow**: 밝기 보존·미소 이동·공간적 일관성 세 가정으로 특징점의 이동을 수식으로 계산
- **Dense Optical Flow**: 특징점이 아닌 모든 픽셀의 움직임을 계산해 장면 전체를 이해
- **RAFT**: 사람이 설계한 가정 대신 데이터로 학습한 신경망이 반복 정제를 통해 Optical Flow를 예측

**핵심 체크리스트**

- [ ] 추적 알고리즘의 발전은 결국 "무엇을 기준으로, 어떻게 비교할 것인가"를 계속 바꿔온 과정이다
- [ ] 템플릿 매칭은 픽셀 전체를 그대로 비교하므로 조명·크기 변화에 취약하다
- [ ] 모서리(Corner)가 좋은 특징점인 이유는 모든 방향에서 뚜렷하게 구별되기 때문이다(구멍 문제 회피)
- [ ] MeanShift는 색상 분포의 무게중심을 쫓지만 윈도우 크기가 고정돼 있다
- [ ] CamShift는 2차 모멘트로 윈도우 크기·방향을 매 프레임 갱신해 이 한계를 보완한다
- [ ] Lucas-Kanade의 3가지 가정(밝기 보존·미소 이동·공간적 일관성)은 미지수보다 방정식을 더 많이 만들어 최소제곱법으로 풀 수 있게 해준다
- [ ] Dense Optical Flow는 모든 픽셀을 계산해 배경까지 포함한 전체 움직임을 파악한다
- [ ] RAFT는 4D Correlation Volume과 GRU 반복 정제로, 사람이 설계한 가정 없이도 예외 상황에 강건하다

결국 객체 추적은 **"이전 프레임의 이 지점이, 다음 프레임의 어디로 갔는가"**라는 단순한 질문에서 출발해, 비교 대상을 점점 정교하게 바꿔가며 마침내 **데이터로 학습한 모델이 그 답을 대신 찾아주는** 단계까지 도달한 과정이라고 볼 수 있습니다.

실무에서는 상황에 맞는 선택이 중요합니다. 실시간성이 중요하고 움직임이 제한적이라면 Lucas-Kanade 같은 전통적 방식이, 정확도가 최우선이고 GPU 여유가 충분하다면 RAFT 같은 딥러닝 기반 방식이 더 적합합니다.

---

*이 글은 OpenCV 스터디 4주차에서 다룬 객체 추적(템플릿 매칭·특징점 매칭·MeanShift·CamShift·Optical Flow·RAFT) 내용을 정리한 기술 블로그입니다.*
