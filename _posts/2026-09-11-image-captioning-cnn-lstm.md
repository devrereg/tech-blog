---
title: "CNN + LSTM으로 이미지 캡셔닝 모델 만들기 — 원리부터 PyTorch 구현까지"
date: 2026-09-11 16:20:00 +0900
categories: [AI, Deep Learning]
tags: [image-captioning, cnn, lstm, encoder-decoder, resnet, pytorch, transfer-learning, teacher-forcing, beam-search, attention, computer-vision, nlp]
description: "사진을 보여주면 문장으로 설명해주는 이미지 캡셔닝 모델을, CNN 인코더·LSTM 디코더 구조부터 단어 사전 구축, Teacher Forcing 학습, Greedy/Beam Search 추론, Attention까지 PyTorch 코드와 함께 정리했다."
---

사진 한 장을 보여주면 "강아지가 공을 물고 있다"처럼 사람처럼 설명해주는 AI, **이미지 캡셔닝(Image Captioning)**의 원리를 이해하고 PyTorch로 직접 구현해본 과정을 정리합니다. 이 글 하나로 이론(어떻게 작동하는가)부터 실전 코드(어떻게 만드는가)까지 전체 흐름을 따라갈 수 있도록 구성했습니다.

> [전이 학습으로 CIFAR-10 분류하기]({% post_url 2026-08-24-transfer-learning-resnet-vgg %})에서 사전학습된 CNN을 "분류기"로 재사용하는 법을 다뤘다면, 이번 글의 CNN 인코더는 같은 사전학습 CNN을 "이미지를 숫자로 요약하는 도구"로 재사용합니다. 그리고 [텍스트 전처리부터 DistilBERT 파인튜닝까지]({% post_url 2026-08-28-nlp-preprocessing-to-distilbert-sentiment %})에서 다룬 텍스트→숫자 변환 파이프라인이 여기서도 그대로 등장합니다.

## 📌 목차

1. [이미지 캡셔닝이란](#what-is-image-captioning)
2. [전체 구조: Encoder-Decoder](#encoder-decoder)
3. [CNN 인코더 구현 — 이미지를 벡터로](#cnn-encoder)
4. [단어 사전(Vocabulary) 만들기](#vocabulary)
5. [LSTM 디코더 구현 — 벡터를 문장으로](#lstm-decoder)
6. [학습: Teacher Forcing & Cross-Entropy Loss](#training)
7. [추론 전략: Greedy Search vs Beam Search](#inference-strategy)
8. [Attention — 어디를 보고 말하나 (심화)](#attention)
9. [성능 평가 지표](#metrics)
10. [Dataset & DataLoader — 실전 데이터 파이프라인](#dataset)
11. [새 이미지에 캡션 달아보기](#predict)
12. [마무리](#summary)

---

## 1. 이미지 캡셔닝이란 {#what-is-image-captioning}

이미지 캡셔닝은 **시각 정보를 이해하는 능력(Computer Vision)**과 **그것을 언어로 표현하는 능력(NLP)**이 결합된 대표적인 멀티모달(Multi-modal) AI 기술입니다.

![사진을 이미지 캡셔닝 모델(CNN+LSTM)에 입력하면 "A dog is running with a ball on the grass."라는 문장을 출력한다는 개념도. CNN을 눈, LSTM을 입에 비유해 설명한다](/assets/img/posts/image-captioning/01_concept_overview.svg)

*사람이 사진을 보면 "아, 강아지가 마루에 앉아있네"라고 즉시 설명할 수 있다. 이걸 AI가 대신 하는 게 이미지 캡셔닝이다*

이걸 가능하게 하려면 두 가지 능력이 결합돼야 합니다.

| 능력 | 역할 | 비유 |
|---|---|---|
| CV (Computer Vision) | 이미지를 "보고" 이해하기 | 눈 |
| NLP (자연어처리) | 이해한 내용을 "말로" 표현하기 | 입 |

이 조합이 바로 이미지 캡셔닝의 핵심 구조인 **CNN(눈) + LSTM(입)**입니다.

---

## 2. 전체 구조: Encoder-Decoder {#encoder-decoder}

이미지 캡셔닝 모델은 크게 두 부분으로 나뉩니다.

```
이미지 → [Encoder: CNN] → 특징 벡터(Feature Vector) → [Decoder: LSTM] → 문장
```

![이미지가 CNN 인코더를 거쳐 특징 벡터로 압축되고, 그 벡터가 LSTM 디코더에 입력되어 문장이 생성되는 5단계 파이프라인 다이어그램](/assets/img/posts/image-captioning/02_encoder_decoder_pipeline.svg)

*인코더가 "무엇이 보이는지"를 압축하면, 디코더가 그것을 문장으로 풀어낸다*

- **인코더(Encoder)**: 입력 이미지를 받아 압축된 특징 벡터로 변환. 주로 **CNN**(Convolutional Neural Network)이 사용되며, 실무에서는 사전학습된 **ResNet, VGG** 등을 활용
- **디코더(Decoder)**: 그 벡터를 받아 자연어 시퀀스(단어들의 나열)를 생성. 주로 **LSTM**(Long Short-Term Memory)이 사용됨

LSTM은 매 순간 "직전에 만든 단어"를 다시 입력받아 다음 단어를 예측하는 **순차적(auto-regressive) 생성** 방식으로 동작합니다. 마치 이어말하기 게임처럼요.

---

## 3. CNN 인코더 구현 — 이미지를 벡터로 {#cnn-encoder}

CNN은 이미지의 픽셀에서 점점 더 추상적인 시각 패턴(선 → 질감 → 사물 형태)을 뽑아냅니다. 처음부터 학습시키지 않고, ImageNet으로 이미 학습된 **ResNet**을 가져다 쓰는 **Transfer Learning**을 사용합니다.

![CNN이 얕은 층에서는 선·모서리·색상 같은 저수준 패턴을, 중간 층에서는 질감·패턴·부분 형태를, 깊은 층에서는 강아지 같은 사물 전체 형태와 의미를 학습한다는 3단계 특징 추출 시각화](/assets/img/posts/image-captioning/03_cnn_feature_hierarchy.svg)

*층이 깊어질수록 픽셀 → 질감 → 사물로, 점점 더 추상적인 패턴을 학습한다*

```python
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        # 사전 학습된 ResNet-50 로드 (ImageNet으로 이미 학습된 가중치)
        # 입력 이미지 shape 예시: (Batch, 3, 224, 224)
        resnet = models.resnet50(pretrained=True)

        # ResNet의 원래 마지막 층(fc)은 (2048 -> 1000) 분류용 선형층인데,
        # 우리는 "분류"가 아니라 "특징 추출"만 필요하므로 이 층을 제거함
        modules = list(resnet.children())[:-1]   # 마지막 fc 층만 제외한 나머지 레이어들
        self.resnet = nn.Sequential(*modules)    # 특징 추출 전용 모듈로 재구성

        # ResNet이 뽑아낸 특징(2048차원)을 LSTM과 호환되는 embed_size 차원으로 변환
        # resnet.fc.in_features == 2048 (ResNet-50 기준)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

        # 학습을 안정시키기 위한 배치 정규화
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # images shape: (Batch, 3, 224, 224)  ← 한 배치에 여러 장의 이미지

        # ResNet 본체는 이미 학습이 끝났으므로 gradient 계산을 하지 않음
        # (Transfer Learning: 특징 추출기는 그대로 재사용, 새로 학습 X)
        # eval()까지 호출해야 내부 BatchNorm의 running_mean/var도
        # 학습 중에 갱신되지 않고 완전히 고정된다 (no_grad만으로는 막을 수 없음)
        self.resnet.eval()
        with torch.no_grad():
            features = self.resnet(images)
            # features shape: (Batch, 2048, 1, 1)  ← 채널 2048짜리 1x1 특징맵

        # (Batch, 2048, 1, 1) -> (Batch, 2048) 로 평탄화(flatten)
        features = features.reshape(features.size(0), -1)

        # 2048차원 -> embed_size 차원으로 변환 + 정규화
        features = self.bn(self.linear(features))
        # 최종 출력 shape: (Batch, embed_size)  ← 이게 LSTM에 전달될 "이미지 요약본"

        return features
```

**핵심 포인트**: ResNet 본체는 `torch.no_grad()`로 감싸서 고정(freeze)하고, 그 위에 새로 붙인 `linear`, `bn` 레이어만 우리 데이터에 맞게 학습시킵니다.

---

## 4. 단어 사전(Vocabulary) 만들기 {#vocabulary}

컴퓨터는 텍스트를 직접 이해할 수 없고 숫자만 계산할 수 있습니다. 그래서 문장을 LSTM에 넣기 전에 **단어를 숫자로 바꾸는 작업**이 필요합니다.

![문장 "A dog is running."이 <start> a dog is running . <end> 형태로 토큰 분리된 뒤, 사전 조회를 거쳐 [1, 15, 234, 8, 892, 7, 2] 정수 인덱스 시퀀스로 변환되는 3단계 흐름도](/assets/img/posts/image-captioning/04_tokenization_flow.svg)

*&lt;start&gt;·&lt;end&gt;·&lt;pad&gt;·&lt;unk&gt; 특수 토큰으로 문장을 정수 시퀀스로 변환*

이 변환에는 특수 토큰 4종 세트가 쓰입니다.

| 토큰 | 역할 |
|---|---|
| `<start>` | 문장 시작 알림 |
| `<end>` | 문장 종료 알림 |
| `<pad>` | 길이 맞춤용 (배치 처리를 위해 짧은 문장 뒤를 채움) |
| `<unk>` | 사전에 없는 모르는 단어 처리 |

```python
from collections import Counter

# --- 단어 사전(Vocabulary) 만들기 ---
special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]

word_counter = Counter()   # {단어: 등장횟수}를 누적해서 세는 자료구조

for caption in all_captions:                # 전체 캡션 문장들을 순회
    for word in caption.split():            # 문장을 공백 기준으로 단어 분리
        word_counter[word] += 1             # 해당 단어의 등장 횟수 +1

# 등장 빈도가 너무 낮은(min_freq 미만) 단어는 제외
# → 희귀 단어는 임베딩이 제대로 학습되지 않고, 사전만 불필요하게 커짐
min_freq = 3
vocab_words = [w for w, cnt in word_counter.items() if cnt >= min_freq]

# 최종 사전: 특수 토큰(4개) + 일반 단어들
idx2word = special_tokens + sorted(vocab_words)   # 인덱스 -> 단어
word2idx = {w: i for i, w in enumerate(idx2word)}  # 단어 -> 인덱스

vocab_size = len(idx2word)   # LSTM의 최종 출력 크기와 직결되는 값
```

---

## 5. LSTM 디코더 구현 — 벡터를 문장으로 {#lstm-decoder}

LSTM(Long Short-Term Memory)은 시퀀스 데이터를 처리하는 RNN 계열 모델로, 이전 단어의 정보를 기억하면서 다음 단어를 예측합니다.

### 5-1. 초기화 (`__init__`)

```python
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        # 단어 인덱스(정수) -> 밀집 벡터(embed_size 차원)로 변환
        # 입력: vocab_size개의 단어 중 하나 (정수), 출력: embed_size 차원 벡터
        self.embed = nn.Embedding(vocab_size, embed_size)

        # 시퀀스를 처리하는 LSTM
        # 입력: embed_size 차원, 은닉 상태: hidden_size 차원
        # batch_first=True → 텐서 shape이 (Batch, Seq_len, Feature) 순서
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # LSTM의 은닉 상태를 "사전에 있는 모든 단어의 확률(로짓)"로 변환
        # 입력: hidden_size, 출력: vocab_size (단어 하나하나에 대한 점수)
        self.linear = nn.Linear(hidden_size, vocab_size)
```

### 5-2. 순전파 (`forward`) — 학습용, Teacher Forcing 적용

```python
    def forward(self, features, captions):
        # features shape: (Batch, embed_size)          ← CNN이 만든 이미지 특징
        # captions shape: (Batch, seq_len)              ← 정답 캡션 (숫자 인덱스)

        # 캡션의 마지막 토큰(<end>)은 입력으로 쓰이지 않음
        # (<end> 다음엔 예측할 단어가 없으므로 입력에서 제외)
        captions = captions[:, :-1]
        # captions shape: (Batch, seq_len - 1)

        # 캡션(정답 단어들)을 임베딩 벡터로 변환 — Teacher Forcing의 핵심
        # (모델이 이전에 뭘 예측했든 상관없이, 항상 "진짜 정답"을 다음 입력으로 사용)
        embeddings = self.embed(captions)
        # embeddings shape: (Batch, seq_len - 1, embed_size)

        # 이미지 특징 벡터를 시퀀스의 "맨 앞 입력"으로 취급하기 위해 차원 추가
        # (Batch, embed_size) -> (Batch, 1, embed_size)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        # inputs shape: (Batch, 1 + (seq_len - 1), embed_size) = (Batch, seq_len, embed_size)
        # → 시퀀스 맨 앞이 "이미지 요약 정보", 그 뒤로 캡션 단어들이 이어지는 구조

        # LSTM 통과: 각 타임스텝마다 문맥 정보를 담은 은닉 상태 계산
        hiddens, _ = self.lstm(inputs)
        # hiddens shape: (Batch, seq_len, hidden_size)

        # 각 타임스텝의 은닉 상태를 "다음 단어 확률(로짓)"로 변환
        outputs = self.linear(hiddens)
        # outputs shape: (Batch, seq_len, vocab_size)

        return outputs
```

### 5-3. 캡션 생성 (`sample`) — 추론용, Greedy Search

```python
    def sample(self, features, states=None):
        # 이 구현은 batch_size=1(이미지 1장씩 추론)을 가정한다.
        # 여러 이미지를 한 번에 생성하려면 predicted.item() 대신
        # predicted.tolist() 등으로 배치 전체를 다뤄야 한다.
        sampled_ids = []   # 생성된 단어 인덱스들을 순서대로 저장할 리스트

        # 이미지 특징을 LSTM의 첫 입력으로 사용
        # (Batch, embed_size) -> (Batch, 1, embed_size)
        inputs = features.unsqueeze(1)

        for i in range(20):   # 최대 20단어까지 생성
            # 현재 입력과 이전 상태(states)를 LSTM에 통과
            hiddens, states = self.lstm(inputs, states)
            # hiddens shape: (Batch, 1, hidden_size)

            # 다음 단어에 대한 확률(로짓) 계산
            outputs = self.linear(hiddens.squeeze(1))
            # outputs shape: (Batch, vocab_size)

            # 사전 전체 단어 중 확률이 가장 높은 단어의 인덱스 선택 (Greedy Search)
            _, predicted = torch.max(outputs, dim=1)
            # predicted shape: (Batch,)  ← 각 배치 샘플마다 예측된 단어 인덱스 1개

            sampled_ids.append(predicted.item())

            # 방금 예측한 단어를 다음 스텝의 입력으로 재사용 (정답이 없으니 자기 예측을 씀)
            inputs = self.embed(predicted).unsqueeze(1)
            # inputs shape: (Batch, 1, embed_size)

        return sampled_ids
```

**forward vs sample 차이**

| | `forward` (학습) | `sample` (추론) |
|---|---|---|
| 입력 | 이미지 특징 + **정답 캡션 전체** | 이미지 특징만 |
| 다음 스텝 입력 | 항상 **정답 단어** (Teacher Forcing) | 직전에 **자기가 예측한 단어** |
| 처리 방식 | 문장 전체를 한 번에 병렬 계산 | 한 단어씩 반복(loop) 생성 |

---

## 6. 학습: Teacher Forcing & Cross-Entropy Loss {#training}

학습 사이클은 아래 5단계로 반복됩니다.

![입력(t-1)에서 모델 예측(t)을 거쳐 정답과 비교하고 Loss를 계산한 뒤 가중치를 업데이트(Backprop)하고, 다시 다음 배치의 입력으로 순환하는 5단계 학습 사이클 다이어그램. Teacher Forcing에 의해 입력은 항상 실제 정답이라는 설명이 붙어 있다](/assets/img/posts/image-captioning/05_training_cycle.svg)

*Teacher Forcing: 항상 "실제 정답"을 다음 스텝 입력(t-1)으로 사용*

```
입력(t-1) → 모델 예측(t) → 정답과 비교 → Loss 계산 → 가중치 업데이트(Backprop)
```

**Teacher Forcing (교사 강요)**
학습 초기에는 모델이 서툴러서 예측이 자주 틀립니다. 모델이 만든 (틀린) 단어를 그대로 다음 입력으로 쓰면 한 번 틀린 게 계속 이어지는 악순환이 생길 수 있어요. 그래서 학습 시에는 모델 예측과 상관없이 **항상 실제 정답을 다음 입력으로 넣어줍니다.** (위 `forward` 코드가 정확히 이 방식)

**Cross-Entropy Loss**
모델이 예측한 단어 확률 분포와 실제 정답(One-hot vector) 사이의 차이를 계산하는 손실 함수입니다.

```python
# pad_idx: <pad> 토큰의 인덱스
# ignore_index=pad_idx → 패딩된 부분은 손실 계산에서 제외 (의미 없는 자리이므로)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

optimizer = torch.optim.AdamW(
    list(decoder.parameters())          # 디코더는 전부 새로 학습
    + list(encoder.linear.parameters())  # 인코더에서 새로 추가한 층만 학습
    + list(encoder.bn.parameters()),
    lr=0.001
)
```

```python
for epoch in range(num_epochs):
    for images, captions, lengths in dataloader:
        # images shape:   (Batch, 3, 224, 224)
        # captions shape: (Batch, seq_len)   ← <start> ... <end> <pad> ... 포함

        features = encoder(images)                  # (Batch, embed_size)
        outputs = decoder(features, captions)        # (Batch, seq_len, vocab_size)

        # 모델 예측(outputs)과 실제 정답(captions)의 위치를 맞춰서 비교
        # 이미지 특징이 맨 앞에 붙어 한 칸 밀렸으므로 인덱스를 정렬
        loss = criterion(
            outputs[:, 1:, :].reshape(-1, vocab_size),  # (Batch*(seq_len-1), vocab_size)
            captions[:, 1:].reshape(-1)                  # (Batch*(seq_len-1),)
        )

        optimizer.zero_grad()   # 이전 gradient 초기화
        loss.backward()          # 역전파로 gradient 계산
        optimizer.step()         # 파라미터 업데이트
```

---

## 7. 추론 전략: Greedy Search vs Beam Search {#inference-strategy}

문장을 만들 때 **다음 단어를 어떻게 고르는지**에도 전략이 있습니다.

![Greedy Search는 매 단계 확률이 가장 높은 단어 1개만 선택해 경로 하나를 추적하는 반면, Beam Search(K=2)는 매 단계 상위 K개의 후보 경로를 동시에 유지하다가 가지치기하는 탐색 트리 비교도](/assets/img/posts/image-captioning/06_greedy_vs_beam_search.svg)

| | Greedy Search | Beam Search |
|---|---|---|
| 매 단계 고려하는 후보 수 | 1개 | K개 |
| 속도 | 빠름 | 느림 (K배 비용) |
| 품질 | 국소 최적해에 갇힐 수 있음 | 대체로 더 자연스러움 |

- **Greedy Search**: 매 순간 확률 1등 단어만 선택 (앞서 구현한 `sample` 메서드 방식)
- **Beam Search**: 상위 K개의 유망한 경로를 동시에 유지하다가, 최종적으로 누적 확률이 가장 높은 문장을 선택

실무에서는 보통 **K = 3~5**가 적절합니다. K가 너무 크면 오히려 무난하고 뻔한 문장만 나오는 역효과가 생길 수 있어요.

---

## 8. Attention — 어디를 보고 말하나 (심화) {#attention}

기본 구조의 한계는, 이미지 전체를 벡터 하나로 압축해서 매 단어를 생성할 때마다 똑같은 정보만 참고한다는 점입니다.

![LSTM이 "dog", "ball", "bites"를 생성할 때마다 이미지의 서로 다른 영역(강아지 부분, 공 부분, 입 부분)에 진한 분홍색으로 표시된 높은 가중치를 부여하는 3단계 Attention 개념도](/assets/img/posts/image-captioning/07_attention_concept.svg)

*진한 분홍 = 해당 단어를 생성할 때 가중치가 높은(주목하는) 이미지 영역*

**Attention**은 사람이 "강아지가 공을 문다"라고 말할 때 "공"을 말하는 순간 시선이 공 쪽으로 이동하는 원리를 모방합니다. LSTM이 단어를 생성할 때마다, 이미지의 각 영역에 대해 동적으로 가중치를 계산해서 필요한 부분에 집중합니다. 정보 손실 문제를 해결하고, 어느 영역을 보고 판단했는지 히트맵으로 시각화할 수 있다는 장점도 있습니다.

---

## 9. 성능 평가 지표 {#metrics}

캡션은 정답이 하나가 아니므로, "얼마나 유사한지"를 수치화하는 지표가 필요합니다.

| 지표 | 핵심 아이디어 | 특징 |
|---|---|---|
| **BLEU** | n-gram 정밀도 | 간단하지만 동의어·문맥 고려 부족 |
| **METEOR** | 정밀도+재현율, 동의어 매칭 | 사람 평가와 상관관계 높음 |
| **CIDEr** | TF-IDF 기반 핵심 단어 일치 | 이미지 캡셔닝 전용, 현재 표준 |

> **실무 팁**: 학습 시 Loss 값만 보지 말고, 검증 데이터셋에 대한 CIDEr 점수를 주기적으로 확인하세요. Loss는 계속 낮아지는데 CIDEr가 정체되거나 떨어진다면 **과적합(Overfitting)** 신호입니다.

---

## 10. Dataset & DataLoader — 실전 데이터 파이프라인 {#dataset}

실제 이미지-캡션 데이터를 모델에 넣으려면, 개별 데이터를 배치(batch)로 묶는 과정이 필요합니다.

```python
import os
from torch.utils.data import Dataset, DataLoader

class CaptionDataset(Dataset):
    def __init__(self, image_dir, image_filenames, captions_dict, transform=None, max_len=20):
        self.image_dir = image_dir                # 이미지 파일들이 들어있는 디렉터리
        self.image_filenames = image_filenames   # 이미지 파일명 리스트
        self.captions_dict = captions_dict        # {이미지파일명: [캡션1, 캡션2, ...]}
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.image_filenames)   # 데이터셋 전체 크기 = 이미지 개수

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        img_path = os.path.join(self.image_dir, img_name)   # 실제 파일 경로 조합

        # 이미지 로드 (224x224로 리사이즈, 텐서 변환, 정규화)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # image shape: (3, 224, 224)

        # 한 이미지당 캡션이 여러 개 있으므로 랜덤하게 하나 선택
        caption = random.choice(self.captions_dict[img_name])
        indices = sentence_to_indices(caption, max_len=self.max_len)
        # indices: [1, 45, 12, 89, 2, 0, 0, ...]  ← <start>...<end>...<pad>

        return image, torch.tensor(indices), len(indices)
        # 반환: (이미지 텐서, 캡션 인덱스 텐서, 실제 길이)


# DataLoader가 자동으로 batch_size개씩 묶어서 반환
# 배치 하나를 꺼내면:
#   images.shape   == (batch_size, 3, 224, 224)
#   captions.shape == (batch_size, max_len)
dataset = CaptionDataset(image_dir, image_filenames, captions_dict, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## 11. 새 이미지에 캡션 달아보기 {#predict}

```python
def predict_caption(image_path):
    # 1. 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    # image shape: (1, 3, 224, 224)  ← 배치 차원 1로 추가

    encoder.eval()   # 평가 모드 (BatchNorm 등이 학습 때와 다르게 동작하도록 설정)
    decoder.eval()

    with torch.no_grad():   # 추론이므로 gradient 계산 불필요
        feature = encoder(image)          # (1, embed_size)
        sampled_ids = decoder.sample(feature)   # 단어 인덱스 리스트, 최대 20개

    # 2. 인덱스를 실제 단어로 변환
    result = []
    for idx in sampled_ids:
        word = idx2word[idx]
        if word == '<end>':
            break   # 문장이 끝났으면 중단
        if word not in ['<start>', '<pad>', '<unk>']:
            result.append(word)

    return ' '.join(result)


# 사용 예시
caption = predict_caption('test_image.jpg')
print(caption)   # 예: "a dog is running on the grass"
```

---

## 마무리 — 전체 그림 정리 {#summary}

```
[이미지]
   ↓ CNN 인코더 (Transfer Learning: ResNet 특징 추출기는 고정, 새 레이어만 학습)
[특징 벡터 (embed_size 차원)]
   ↓ LSTM 디코더 초기 입력으로 사용
[단어 하나씩 순차 생성] ← (심화) Attention으로 매번 다른 이미지 영역에 집중
   ↓ 학습 시: Teacher Forcing + Cross-Entropy Loss (ignore_index=pad_idx)
   ↓ 추론 시: Greedy Search 또는 Beam Search
[<end> 토큰] → 문장 완성!
```

핵심만 다시 정리하면:

1. **CNN(눈) + LSTM(입)** 구조가 이미지 캡셔닝의 뼈대
2. 텍스트는 반드시 **숫자(인덱스)**로 변환해야 컴퓨터가 처리 가능 (`<pad>/<start>/<end>/<unk>` 특수 토큰 필수)
3. 학습 때는 **Teacher Forcing**으로 안정적으로, 추론 때는 **자기 예측을 재사용**하며 문장 생성
4. **Transfer Learning**으로 이미 학습된 CNN을 재사용해 효율적으로 학습
5. 평가는 **CIDEr** 같은 이미지 캡셔닝 전용 지표로 확인

이 구조를 이해하면, 이후 Attention 메커니즘을 직접 구현하거나 Transformer 기반 캡셔닝 모델(BLIP, GIT 등)로 확장하는 것도 자연스럽게 이어갈 수 있습니다.
