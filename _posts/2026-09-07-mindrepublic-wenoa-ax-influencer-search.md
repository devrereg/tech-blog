---
title: "[마인드리퍼블릭] WENOA AX 서비스 — 자연어 기반 인플루언서 검색"
date: 2026-09-07 02:00:00 +0900
categories: [포트폴리오]
tags: [portfolio, mindrepublic, wenoa, aws, ecs-fargate, serverless, fastapi, langgraph, llm, ai-agent, elasticsearch, github-actions]
description: "마인드리퍼블릭 WENOA의 내부 직원용 AX 서비스. 기존 멀티 필터 방식의 인플루언서 검색을, 자연어 요청을 받아 AI Agent가 검색 필터를 자동 생성하고 가중치로 정렬하는 방식으로 바꿨다. 비용 절감을 위해 ECS Fargate 기반 서버리스 아키텍처로 구성했다."
---

> 마인드리퍼블릭에서 운영하는 서비스 **WENOA**의 내부 직원용 **AX(AI Transformation) 서비스** 개발 기록입니다. 기존 멀티 필터 방식의 인플루언서 검색을, **자연어 한 문장**으로 원하는 인플루언서를 찾는 방식으로 전환했습니다.

## 프로젝트 개요

- **회사 / 서비스**: 마인드리퍼블릭 / WENOA
- **프로젝트명**: WENOA 전용 AX 서비스 개발 (자연어 기반 인플루언서 검색)
- **일정**: 2026.03 (2주)
- **기술 스택**: AWS (ECS Fargate, ALB, ECR, Route 53, Secrets Manager, CloudWatch, IAM) · Docker · GitHub Actions · FastAPI · LangGraph

**설계 기준**

- 내부 직원 전용 서비스로 트래픽 규모가 크지 않은 API 서버
- 비용 절감을 위해 서버리스 아키텍처(ECS Fargate 기반) 도입

---

## 아키텍처

![WENOA AX 서비스 아키텍처(2026.03) 다이어그램. CI/CD 파이프라인은 개발자 코드 Push → GitHub Repository → GitHub Actions(Docker Build & Push, GitHub OIDC 기반 IAM Role로 AWS 배포 권한 획득) → ECR → ECS Fargate 배포로 이어진다. 런타임은 내부 직원(사내망/VPN) → Route 53 → ALB → ECS Fargate(FastAPI Task)로 요청이 들어오고, 기존 방식은 멀티 필터 검색 API가 Elasticsearch에 필터 기반 쿼리를 보내 정렬 없는 결과 리스트를 반환한다. 신규 방식은 자연어 검색 API가 LangGraph Agent를 실행해 ① 자연어 파싱 및 필터 자동 생성(LLM, Secrets Manager에서 API Key 조회) → ② 자동 생성 필터로 Elasticsearch 검색 실행 → ③ 가중치 기반 인플루언서 정렬을 거쳐 정렬된 결과 리스트를 반환한다. ECS Task는 CloudWatch로 로그를 전송하고 IAM(ECS Task Role)로 권한을 부여받는다](/assets/img/posts/mindrepublic-wenoa-ax/wenoa_ax_architecture.drawio.png)

*WENOA AX 서비스 아키텍처 (2026.03) — CI/CD(GitHub Actions → ECR → ECS Fargate)와 런타임(FastAPI Task 위에서 기존 멀티 필터 검색 API와 자연어 검색 API가 공존, 후자는 LangGraph Agent가 처리)*

---

## 문제

기존 Elasticsearch 기반 인플루언서 검색은 **멀티 필터 방식**이었다.

- 사용자가 카테고리·팔로워 수·지역·참여율 등 **적합한 검색 조건을 직접 골라 설정**해야 했다.
- 어떤 필터 조합이 원하는 결과를 주는지 사용자가 미리 알아야 해서, 검색 자체에 러닝 커브가 있었다.

---

## 해결

자연어 요청을 입력받아 **AI Agent가 검색 필터를 자동 생성**하고, 가중치 기반으로 인플루언서를 정렬해 리스트로 제공하는 기능을 구현했다. Agent는 **LangGraph**로 설계했다.

- **① 자연어 파싱 & 필터 자동 생성** — 사용자의 자연어 요청을 LLM이 해석해 Elasticsearch 검색 필터로 변환한다. (LLM API Key는 Secrets Manager에서 조회)
- **② ES 검색 실행** — 자동 생성된 필터로 Elasticsearch 인플루언서 인덱스를 조회한다.
- **③ 가중치 기반 정렬** — 검색 결과를 요청 의도에 맞춰 가중치로 재정렬해 리스트로 반환한다.
- **인프라** — 내부 직원 전용·저트래픽 특성에 맞춰 **ECS Fargate 기반 서버리스**로 구성하고, GitHub Actions에서 OIDC로 AWS 권한을 받아 ECR → ECS Fargate로 배포한다. 기존 멀티 필터 검색 API와 신규 자연어 검색 API는 동일한 FastAPI Task 위에서 공존한다.

---

## 결과

- 사용자가 **별도 필터 설정 없이 자연어만으로** 원하는 인플루언서를 효율적으로 검색할 수 있게 됐다.
- 기존 멀티 필터 검색을 유지하면서 자연어 검색을 **추가 API로 얹어**, 사용자가 두 방식을 선택할 수 있다.
- 저트래픽 내부 서비스를 ECS Fargate 서버리스로 구성해 상시 운영 비용을 낮췄다.
