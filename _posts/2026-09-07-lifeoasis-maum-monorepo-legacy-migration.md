---
title: "[라이프오아시스] MAUM 운영 안정화 — 멀티 레포 마이크로서비스 Monorepo 통합 및 레거시 마이그레이션"
date: 2026-09-07 07:00:00 +0900
categories: [포트폴리오]
tags: [spring-boot, kotlin, gradle, mysql, dynamodb, redis, grpc, graphql, kubernetes, grafana, monorepo, git-submodule]
description: 여러 프레임워크로 나뉘어 있던 11개 마이크로서비스를 Monorepo로 합치고, Spring Boot 3 + Kotlin으로 마이그레이션한 과정을 정리
---

## 프로젝트 개요

- **회사 / 서비스**: 라이프오아시스 / MAUM
- **프로젝트명**: MAUM 운영 안정화 및 레거시 마이그레이션
- **일정**: 2023.05 ~ 2025.05 (2년)
- **역할**: 백엔드 개발 (백엔드 3인, Monorepo 구조 설계 및 마이그레이션)
- **기술 스택**
  - Backend: Spring Boot 3 (Kotlin) · Gradle 멀티모듈 · gRPC · GraphQL (API Gateway)
  - Legacy: NestJS · Python (Django, FastAPI) · Go
  - Database: MySQL · DynamoDB · Redis
  - Infra: Kubernetes · Grafana · Git Submodule

---

## 문제

MAUM은 11개의 마이크로서비스가 **서비스마다 별도 repo**로 나뉘어 있었고, **NestJS, Django, FastAPI, Go** 등 각기 다른 프레임워크로 만들어져 있었다. 이걸 백엔드 3명이 모두 관리해야 했다.

기능 하나를 개발하려면 관련 repo를 여러 개 동시에 띄워야 했다. repo마다 만든 사람이 달라서, 다른 서비스의 로직을 써야 할 때마다 담당자에게 확인하는 커뮤니케이션 비용이 컸다.

프레임워크가 제각각이다 보니 한 사람이 전체를 유지보수하기 어려웠고, 이 스택을 다 다룰 수 있는 사람을 구하기도 힘들어 **MAUM 담당 개발자 채용**에도 어려움이 있었다.

---

## 해결

흩어진 repo를 **하나의 Monorepo로 합쳐** 한 곳에서 개발하고 테스트할 수 있게 했다. 그리고 각 서비스를 **Spring Boot 3 + Kotlin**으로 하나씩 옮기면서 Gradle 멀티모듈로 구성했다.

옮기는 과정에서 repo마다 중복돼 있던 공통 함수, 테이블 엔티티, 라이브러리는 **공통 모듈로 분리**해 여러 서비스가 같이 쓰도록 했다. 코드를 다시 보면서 발견한 **Slow Query와 작은 버그들**도 함께 수정했다.

**설계 기준**

- 프로젝트 하나만 열면 관련 서비스를 모두 빌드·실행·테스트할 수 있어야 함
- 운영 중인 서비스이므로 옮기는 동안에도 서비스가 멈추면 안 됨
- 기존 서비스와 새 서비스가 함께 돌아가는 동안 API가 깨지면 안 됨
- 급하게 수정할 일이 생기면 기존 방식대로도 고칠 수 있어야 함
- 적은 인원으로 유지보수할 수 있고, 채용이 쉬운 스택 하나로 모아야 함

## 아키텍처

![MAUM 멀티 레포 → Monorepo 통합 및 Spring Boot 3 + Kotlin 마이그레이션 아키텍처](/assets/img/posts/maum-monorepo-migration/maum_legacy_migration_architecture.png)

- **Before**: 11개 서비스가 각각 별도 repo로 존재 (NestJS · Django · FastAPI · Go)
- **After**: 하나의 Monorepo(MAUM) 안에서 Spring Boot 3 + Kotlin Gradle 멀티모듈로 구성. 공통 함수, 테이블 엔티티, 라이브러리는 공통 모듈로 빼고 루트 Gradle에서 의존관계를 관리
- **미전환 서비스**: 기존 repo에서 그대로 운영해 급한 수정은 기존 방식대로 처리하고, 전환 후 안정화되면 기존 repo 제거
- **데이터 / 통신**: MySQL · DynamoDB · Redis, gRPC · GraphQL API Gateway는 기존 방식 그대로 사용하고, Grafana로 모니터링

작업은 **Monorepo 통합 → 서비스별 순차 마이그레이션 및 공통 모듈 분리 → 안정화 후 기존 repo 제거** 순서로 진행했다.

**Monorepo를 선택한 이유**: 서비스가 많은 것보다, 적은 인원이 여러 repo를 오가며 환경을 맞추는 게 더 큰 문제였다. 서비스는 모듈로 나눠둔 채 저장소만 하나로 합치면, MSA 구조는 유지하면서 개발 환경은 하나로 쓸 수 있었다.

**순차 마이그레이션을 선택한 이유**: 운영 중인 서비스를 한 번에 바꾸면 장애 위험이 크다. 서비스 단위로 하나씩 옮기고, 서비스 간 통신(gRPC · GraphQL)과 DB는 그대로 둬서 기존 서비스와 새 서비스가 같이 돌아가는 동안에도 API가 깨지지 않게 했다.

**Spring Boot 3 + Kotlin을 선택한 이유**: 목표는 여러 개로 흩어진 기술 스택을 하나로 모으는 것이었다. 국내 개발자 풀이 가장 넓은 Spring Boot를 선택해 채용 문제를 해결하고자 했고, 언어는 Java보다 가독성이 좋은 Kotlin을 선택했다.

---

## 결과

- 여러 repo를 오가던 비효율이 없어지고, 한 곳에서 개발·테스트 가능해짐
- 중복 구현돼 있던 함수와 엔티티를 공통 모듈로 합쳐 한 곳만 수정하면 되도록 개선
- 스택 통일로 **MAUM 담당 백엔드 개발자 채용 성공**
- 11개 서비스 중 **6개 전환 완료, 1개 진행 중** (진행 중 포함 약 60%)
- Slow Query 개선 및 작은 버그들 수정

---

## 회고

- - 기존 로직을 이해하려고 담당자들과 이야기하면서, "왜 이렇게 개발됐는지"에 대한 히스토리를 알게 된 게 즐거웠다.
- 여러 서비스를 거치는 요청을 한 번에 추적할 수 있는 모니터링 기능을 같이 만들었다면 장애 대응이 더 편했을 것 같다.
- MSA에서는 장애 발생 시 데이터 정합성을 맞추는 과정이 꽤 힘들다는 것을 알았다.