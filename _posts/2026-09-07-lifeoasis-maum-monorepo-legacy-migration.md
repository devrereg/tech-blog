---
title: "[라이프오아시스] MAUM 운영 안정화 및 레거시 마이그레이션 — 분산된 멀티 레포 마이크로서비스를 Monorepo로 통합하고 Spring Boot 3 + Kotlin으로 순차 마이그레이션"
date: 2026-09-07 07:00:00 +0900
categories: [포트폴리오]
tags: [portfolio, lifeoasis, maum, monorepo, microservices, legacy-migration, spring-boot, kotlin, mysql, dynamodb, redis, grpc, graphql, kubernetes, k8s, grafana, slow-query]
description: "라이프오아시스 MAUM의 2년(2023.05~2025.05) 운영 안정화·레거시 마이그레이션 기록. 여러 repo로 분산 운영되던 마이크로서비스를 하나의 Monorepo로 통합해 소수 인력의 개발·테스트 비효율을 없애고, 서로 다른 프레임워크로 구현돼 있던 서비스를 채용 시장 수요가 가장 높은 Spring Boot 3 + Kotlin으로 순차 마이그레이션하면서 Slow Query 개선 등 성능 최적화를 병행했다."
---

> 라이프오아시스에서 운영하는 서비스 **MAUM**의 **운영 안정화 및 레거시 마이그레이션** 기록입니다. 백엔드 인력 2명이 여러 repo로 분산된 마이크로서비스를 관리하던 구조를 **하나의 Monorepo로 통합**하고, 마이크로서비스별로 제각각이던 프레임워크를 **Spring Boot 3 + Kotlin으로 순차 마이그레이션**하면서 성능 최적화까지 함께 진행한 2년간의 과정을 정리합니다.

## 프로젝트 개요

- **회사 / 서비스**: 라이프오아시스 / MAUM
- **프로젝트명**: 운영 안정화 및 레거시 마이그레이션
- **일정**: 2023.05 ~ 2025.05 (2년)
- **기술 스택**: Kubernetes (K8s) · Spring Boot 3 (Kotlin) · MySQL · DynamoDB · Redis · GraphQL · gRPC · Grafana

**설계 기준**

- 여러 repo로 분산 운영되던 마이크로서비스를 **Monorepo로 통합**
- 다양한 프레임워크로 구현된 마이크로서비스를 **Spring Boot 3 + Kotlin으로 마이그레이션**

---

## 아키텍처

![MAUM 멀티 레포 → Monorepo 통합 및 Spring Boot 3 + Kotlin 마이그레이션 아키텍처(2023.05~2025.05) 다이어그램. 기술 스택은 K8s, Spring Boot3 + Kotlin, MySQL, DynamoDB, Grafana, GraphQL, gRPC, Redis. 상단 통합 전(Before): 마이크로서비스마다 별도의 Git Repository가 존재하고 각 repo가 서로 다른 언어·프레임워크로 구현돼 있다 — MSA #1 Repo(Node.js/Express), MSA #2 Repo(Django/Python), MSA #3 Repo(Ruby on Rails), MSA #4 Repo(Go/Gin). 왼쪽의 '백엔드 인력 2명'이 이 여러 repo를 점선 화살표로 이어 개발·관리(비효율)하며, 기능 개발·테스트 시 여러 repo를 동시에 띄워야 하는 비효율과 프레임워크 혼재로 인한 채용 어려움이 있다. 각 repo에서 아래쪽 K8s Cluster 안의 Monorepo(MAUM)로 '마이그레이션' 화살표가 향한다. 하단 통합 후(After): K8s Cluster 안의 하나의 Monorepo(MAUM)에 MSA #1~#4 모듈이 디렉터리로 들어가 있다 — MSA #1(Spring Boot3 + Kotlin, 완료 ✅), MSA #2(Spring Boot3 + Kotlin, 완료 ✅), MSA #3(마이그레이션 중, 60%), MSA #4(마이그레이션 예정). 왼쪽의 '백엔드 개발자(신규 채용 성공)'가 '단일 환경에서 개발·테스트'한다. Monorepo 아래에는 데이터/통신 계층으로 MySQL(Slow Query 개선), DynamoDB, Redis, gRPC / GraphQL API Gateway가 연동되고, 이들의 '메트릭 수집'이 맨 아래 Grafana(통합 모니터링)로 모인다. 하단 결과: MAUM 담당 백엔드 개발자 채용 성공, 레거시 60% 마이그레이션 완료, 성능 개선(Slow Query 등) 동시 달성](/assets/img/posts/maum-monorepo-migration/maum_legacy_migration_architecture.drawio.png)

*MAUM 레거시 마이그레이션 구조 (2023.05~2025.05) — Node.js/Express·Django/Python·Ruby on Rails·Go/Gin 등 서로 다른 프레임워크로 분리 운영되던 마이크로서비스(MSA #1~#4) repo를 K8s Cluster 위의 하나의 Monorepo(MAUM)로 합쳐 단일 환경에서 개발·테스트하도록 만들고, 각 모듈을 Spring Boot 3 + Kotlin으로 순차 마이그레이션했다(MSA #1·#2 완료, #3 진행 중 60%, #4 예정). 데이터/통신 계층은 MySQL·DynamoDB·Redis·gRPC/GraphQL API Gateway를 공유하고 Grafana로 통합 모니터링하며, MySQL Slow Query 개선 등 성능 최적화를 병행했다.*

전체 작업은 **① 멀티 레포 → Monorepo 통합 → ② 모듈 단위 Spring Boot 3 + Kotlin 순차 마이그레이션 → ③ 마이그레이션과 함께 성능 최적화** 순으로 진행됐다.

**1. 멀티 레포 → Monorepo 통합**

- 마이크로서비스마다 흩어져 있던 저장소를 하나의 Monorepo로 합치고, 각 서비스를 모듈(디렉터리) 단위로 배치했다.
- 개발자가 **단일 프로젝트만 열면** 관련 서비스를 전부 빌드·실행·테스트할 수 있도록 구조를 정리했다.

**2. 모듈 단위 Spring Boot 3 + Kotlin 마이그레이션**

- Node.js/Express, Django/Python, Ruby on Rails, Go/Gin 등 서로 다른 프레임워크로 구현된 모듈(MSA #1~#4)을 **Spring Boot 3 + Kotlin**으로 하나씩 순차 마이그레이션했다.
- 운영 중인 서비스이므로 전체를 한 번에 바꾸지 않고, 모듈 경계 단위로 옮기면서 기존 레거시와 신규 구현이 공존하도록 했다. (MSA #1·#2 완료, #3 진행 중 60%, #4 예정)
- 서비스 간 통신은 gRPC·GraphQL(API Gateway)로, 데이터 계층은 MySQL·DynamoDB·Redis로 유지해 마이그레이션 중에도 인터페이스가 깨지지 않게 했다.

**3. 성능 최적화 병행**

- 마이그레이션으로 코드를 다시 들여다보는 시점에 맞춰 **Slow Query를 함께 개선**했다.
- K8s 배포·Grafana 모니터링 위에서 개선 전후 지표를 확인하며 진행했다.

---

## 문제

### 1. 소수 인력의 다수 레포 관리로 인한 개발 비효율

- 백엔드 인력 **2명**이 여러 repo로 분산된 마이크로서비스를 모두 관리해야 했다.
- 기능 하나를 개발·테스트하려면 관련된 **여러 repo를 동시에 clone·실행**해야 했고, repo마다 실행 방법·설정·의존성이 달라 준비 과정 자체가 비용이었다.
- repo가 늘어날수록 소수 인력이 감당해야 하는 유지보수 범위가 넓어졌다.

### 2. 다양한 프레임워크 혼재로 인한 채용 및 유지보수 어려움

- 마이크로서비스별로 **서로 다른 프레임워크**가 사용되고 있었다. (Node.js/Express, Django/Python, Ruby on Rails, Go/Gin 등)
- MAUM 담당 개발자를 채용하려 해도, 여러 프레임워크를 모두 다룰 수 있는 적합한 인력을 찾기 어려웠다.
- 프레임워크마다 관례·빌드·테스트 방식이 달라, 한 사람이 전체를 유지보수하는 부담이 컸다.

---

## 해결

### 1. 전체 마이크로서비스를 Monorepo로 통합

- MAUM의 마이크로서비스를 **하나의 Monorepo로 통합**하고, 각 서비스를 모듈 단위로 배치했다.
- 단일 프로젝트 환경에서 관련 서비스를 한 번에 띄워 **개발·테스트가 가능**하도록 구조를 개선했다. 기능 개발 시 여러 repo를 오가며 맞추던 작업이 사라졌다.
- 저장소가 하나로 모이면서, 소수 인력이 전체 코드베이스를 한눈에 파악하고 유지보수할 수 있게 됐다.

### 2. Spring Boot 3 + Kotlin으로 순차 마이그레이션 + 성능 최적화

- Monorepo 통합과 함께, 각 마이크로서비스를 **채용 시장에서 수요가 가장 높은 Spring Boot 3 기반**으로 순차 마이그레이션했다.
- 운영 리스크를 줄이기 위해 모듈 단위로 나눠 옮기고, 마이그레이션이 끝난 모듈부터 표준 스택으로 수렴시켰다.
- 마이그레이션 과정에서 코드를 다시 검토하는 김에 **Slow Query 개선 등 성능 최적화를 병행**해, 구조 개선과 성능 개선을 한 번의 작업으로 처리했다.

---

## 결과

### 1. 소수 인력의 다수 레포 관리로 인한 개발 비효율

- 다중 repo 운영으로 인한 개발 비효율이 제거됐다.
- 단일 프로젝트 환경에서 개발·테스트가 가능해져, **소수 인력으로도 효율적인 유지보수**가 가능해졌다.

### 2. 다양한 프레임워크 혼재로 인한 채용 및 유지보수 어려움

- 표준 스택(Spring Boot 3)으로 수렴하면서 **MAUM 담당 백엔드 개발자 채용에 성공**했다.
- 기존 레거시의 **약 60% 마이그레이션을 완료**했다.
- Slow Query 개선 등 **서비스 성능 개선까지 동시에 달성**했다.
