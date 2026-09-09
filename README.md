# devrereg's tech blog

[![Build and Deploy](https://github.com/devrereg/tech-blog/actions/workflows/pages-deploy.yml/badge.svg)](https://github.com/devrereg/tech-blog/actions/workflows/pages-deploy.yml)
[![GitHub license](https://img.shields.io/github/license/devrereg/tech-blog.svg?color=blue)](https://github.com/devrereg/tech-blog/blob/main/LICENSE)

> 백엔드 개발 학습 기록과 실무 프로젝트 회고를 모아두는 기술 블로그입니다.

**🔗 [devrereg.github.io/tech-blog](https://devrereg.github.io/tech-blog)**

## 소개

어떤 요구사항이 와도 시스템을 안정적으로 구현할 수 있는 백엔드 개발자를 목표로 합니다.
학습한 내용과 실무에서 마주한 문제·해결 과정을 정리해 아카이빙합니다.

- **관심 분야** — 시스템 아키텍처 · 대규모 트래픽의 안정적 운영 · 동시성 제어
- **주로 다루는 스택** — Kotlin · Spring Boot 3 · MySQL · Redis · Kafka · Kubernetes · Python (FastAPI)

---

## 실무 포트폴리오

문제 상황 → 해결 접근 → 결과 순으로 정리한 프로젝트 회고입니다.

### 마인드리퍼블릭 · WENOA (2025.07 ~ )

인플루언서 마케팅 플랫폼 WENOA의 백엔드 개발.

| 기간 | 프로젝트 | 요약 |
|---|---|---|
| 2026.03 | [자연어 기반 인플루언서 검색 (AX 서비스)](https://devrereg.github.io/tech-blog/posts/mindrepublic-wenoa-ax-influencer-search/) | 멀티 필터 방식 검색을 "자연어 요청 → AI Agent가 검색 필터 자동 생성·가중치 정렬" 방식으로 전환. 비용 절감을 위해 ECS Fargate 기반 서버리스로 구성 |
| 2026.01 | [정기결제 토스페이먼츠 연동](https://devrereg.github.io/tech-blog/posts/mindrepublic-tosspayments-recurring-billing/) | 빌링키 기반 정기결제 연동. 외부 PG와 내부 DB 상태가 어긋나는 정합성 이슈를 보상 트랜잭션(자동 환불) + Slack 실시간 알림으로 대응 |
| 2025.12 | [FE/BE 통합 AI-Native 개발환경 구축](https://devrereg.github.io/tech-blog/posts/mindrepublic-monorepo-ai-native-devenv/) | FE·BE 저장소를 Git Subtree로 Monorepo에 통합해 기능 단위 개발·단일 PR 실현. 폴더별 `CLAUDE.md` 컨텍스트 엔지니어링으로 AI 도구의 토큰 부족 문제 해결 |
| 2025.10 | [유튜브 콘텐츠 실시간 수집](https://devrereg.github.io/tech-blog/posts/mindrepublic-wenoa-youtube-collector/) | 일 20만 건 이상으로 늘어난 수집량에 맞춰 SQS 컨슈머 병렬도·EC2 오토스케일링 조정. MySQL 커넥션 한계와 YouTube Data API 일일 한도(10,000 credit)를 50건 배치 호출로 돌파 |
| 2025.07 | [인플루언서 검색 ETL 파이프라인 구축](https://devrereg.github.io/tech-blog/posts/mindrepublic-wenoa-influencer-search-etl-pipeline/) | 일 1,000개 키워드로 최대 5만 채널을 수집·가공·저장. 채널 설명과 인기 게시물의 이미지·텍스트를 결합한 멀티모달 임베딩으로 텍스트 기반 검색을 벡터 기반 의미 검색으로 전환 |
| — | [서버 장애 알림 처리](https://devrereg.github.io/tech-blog/posts/mindrepublic-error-alert-notification/) | AWS WAF Email 알림 + 전 서버 에러 로그 Slack Bot 실시간 통지 파이프라인 구축. dedup 키 + Redis 억제 창으로 동일 장애 중복 알림 제거 |

### 라이프오아시스 · MAUM (2023.05 ~ 2025.05)

800만 유저 규모 소셜 앱 MAUM의 백엔드 운영·개발.

| 기간 | 프로젝트 | 요약 |
|---|---|---|
| 2025.03 | [근처 친구 추천](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-nearby-friends-recommendation/) | 요청마다 800만 유저 위치정보를 전체 스캔하던 구조를, 3시간 주기 배치가 Redis Geolocation으로 근처 친구 50명을 미리 계산해 유저별 Redis Set에 캐싱하는 구조로 전환 |
| 2024.03 | [토스페이먼츠 웹결제 연동](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-tosspayments-web-payment/) | iOS 인앱결제 30% 수수료를 우회하는 웹결제 플로우 설계. 결제 성공 후 재화 지급 실패에 대비해 Saga 패턴 보상 트랜잭션(자동 환불)과 Slack 알림 구현 |
| 2023–2025 | [Admin/백오피스 개발 및 자동화](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-admin-fulltext-search-promotion-automation/) | 800만 유저 대상 이름·이메일 LIKE 검색(20초+)을 MySQL Fulltext Index(MATCH AGAINST)로 200ms까지 개선. 매월 2~3일씩 걸리던 마케팅 프로모션 세팅을 마케터 셀프서비스 기능으로 자동화 |
| 2023–2025 | [운영 안정화 및 레거시 마이그레이션](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-monorepo-legacy-migration/) | 여러 repo로 분산된 마이크로서비스를 하나의 Monorepo로 통합하고, 이기종 프레임워크로 흩어진 서비스를 Spring Boot 3 + Kotlin으로 순차 마이그레이션하며 Slow Query 개선 등 성능 최적화 병행 |

---

## 학습 기록

### 동시성 제어 — 쿠폰 선착순 발급 시리즈

재고 100개에 2,000명이 몰리는 동일한 조건에서, 락 없는 구현의 한계부터 비관 락 · 낙관 락 · Redis · Kafka까지 단계적으로 발전시키며 정확성과 성능의 트레이드오프를 비교한 6부작.

1. [락 없는 쿠폰 발급 1차 구현과 그 한계](https://devrereg.github.io/tech-blog/posts/coupon-concurrency-1-baseline/)
2. [lost update를 테스트로 재현하기](https://devrereg.github.io/tech-blog/posts/coupon-concurrency-2-reproduce-lost-update/)
3. [비관 락으로 lost update 잡기](https://devrereg.github.io/tech-blog/posts/coupon-concurrency-3-pessimistic-lock/)
4. [낙관 락 발급 경로와 비관 락의 재시도 비용 비교](https://devrereg.github.io/tech-blog/posts/coupon-concurrency-4-optimistic-lock/)
5. [재고 판단을 Redis로 옮기기](https://devrereg.github.io/tech-blog/posts/coupon-concurrency-5-redis-gate/)
6. [발급을 비동기로: Kafka로 DB 저장 떼어내기](https://devrereg.github.io/tech-blog/posts/coupon-concurrency-6-kafka-async/)

### Backend

- [Spring AOP 왜, 언제, 어떻게 사용하는가?](https://devrereg.github.io/tech-blog/posts/spring-aop-basics/) — 분산 락 예제로 정리
- [Flyway란 무엇인가? 데이터베이스 변경 이력 관리와 실무 운영](https://devrereg.github.io/tech-blog/posts/spring-boot-flyway-and-ddl-auto/)
- [Pydantic과 AsyncIO, 손으로 만지며 배운 하루](https://devrereg.github.io/tech-blog/posts/pydantic-asyncio-first-day/)
- [SQLAlchemy 2.0 async, 함정 두 개로 배운 ORM](https://devrereg.github.io/tech-blog/posts/sqlalchemy-2-async-two-pitfalls/)
- [FastAPI, 따로 배운 셋을 하나의 API로 묶기 — 의존성 생명주기를 중심으로](https://devrereg.github.io/tech-blog/posts/fastapi-dependency-lifecycle/)

### AI · Computer Vision

머신러닝/딥러닝 스터디 정리. 전체 목록은 [Archives](https://devrereg.github.io/tech-blog/archives/)에서 볼 수 있습니다.

- **OpenCV 스터디 (4주)** — [디지털 영상의 기초](https://devrereg.github.io/tech-blog/posts/opencv-digital-image-basics/) · [필터·기하학적 변환](https://devrereg.github.io/tech-blog/posts/opencv-geometric-transformation/) · [특징 검출](https://devrereg.github.io/tech-blog/posts/opencv-feature-detection/) · [객체 추적 알고리즘](https://devrereg.github.io/tech-blog/posts/opencv-object-tracking/)
- **딥러닝 기초 · PyTorch** — [딥러닝 기초 완전 정리](https://devrereg.github.io/tech-blog/posts/deep-learning-basics/) · [PyTorch 기본](https://devrereg.github.io/tech-blog/posts/pytorch-basics-introduction/) · [선형 회귀와 학습 루프](https://devrereg.github.io/tech-blog/posts/pytorch-linear-regression-training-loop/)
- **분류 모델** — [이진 분류](https://devrereg.github.io/tech-blog/posts/binary-classification-with-pytorch/) · [BCEWithLogitsLoss vs Sigmoid](https://devrereg.github.io/tech-blog/posts/bce-with-logits-vs-sigmoid/) · [다중 분류](https://devrereg.github.io/tech-blog/posts/multi-class-classification-with-pytorch/) · [MNIST 다중분류와 평가지표](https://devrereg.github.io/tech-blog/posts/pytorch-mnist-multiclass-classification/)
- **CNN · 전이 학습** — [CNN 이미지 분류](https://devrereg.github.io/tech-blog/posts/cnn-image-classification/) · [CNN 완전 정복](https://devrereg.github.io/tech-blog/posts/cnn-convolution-complete-guide/) · [CNN 튜닝·과적합](https://devrereg.github.io/tech-blog/posts/deep-learning-tuning-overfitting/) · [전이 학습 ResNet vs VGG](https://devrereg.github.io/tech-blog/posts/transfer-learning-resnet-vgg/) · [파인튜닝 전략](https://devrereg.github.io/tech-blog/posts/transfer-learning-finetuning-guide/) · [ImageFolder 전이 학습](https://devrereg.github.io/tech-blog/posts/pytorch-imagefolder-transfer-learning/)
- **NLP** — [텍스트 전처리부터 DistilBERT 파인튜닝까지](https://devrereg.github.io/tech-blog/posts/nlp-preprocessing-to-distilbert-sentiment/)

---

## 최근 글

<!-- BLOG-POST-LIST:START -->
- [AI는 새로운 이미지를 어떻게 만들까: 생성형 이미지 AI&lpar;GAN, VAE&rpar; 원리 총정리](https://devrereg.github.io/tech-blog/posts/generative-image-ai-gan-vae/)
- [[라이프오아시스] MAUM Admin/백오피스 개발 및 자동화 — 800만 유저 LIKE 검색을 Fulltext Index로 100배 개선하고, 매월 반복되던 마케팅 프로모션 세팅을 Admin 기능으로 자동화](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-admin-fulltext-search-promotion-automation/)
- [[라이프오아시스] MAUM 인앱결제 수수료 절감을 위한 토스페이먼츠 웹결제 연동 — 결제·재화 지급 정합성을 Saga 보상 트랜잭션으로 확보](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-tosspayments-web-payment/)
- [[라이프오아시스] MAUM 근처 친구 추천 — 800만 유저 위치정보 전체 스캔을 3시간 배치 캐싱으로 전환](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-nearby-friends-recommendation/)
- [[라이프오아시스] MAUM 운영 안정화 및 레거시 마이그레이션 — 분산된 멀티 레포 마이크로서비스를 Monorepo로 통합하고 Spring Boot 3 + Kotlin으로 순차 마이그레이션](https://devrereg.github.io/tech-blog/posts/lifeoasis-maum-monorepo-legacy-migration/)
<!-- BLOG-POST-LIST:END -->

## 연락

- GitHub: [@devrereg](https://github.com/devrereg)
- Email: devrereg@gmail.com
