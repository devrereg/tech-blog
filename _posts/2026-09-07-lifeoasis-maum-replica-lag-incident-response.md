---
title: "[라이프오아시스] MAUM 간헐적 요청 실패 장애 대응 — Read/Write DB 분리 환경의 복제 지연(Replication Lag) 원인 분석"
date: 2026-09-07 13:00:00 +0900
categories: [포트폴리오]
tags: [mysql, read-write-splitting, spring-boot, kotlin]
description: "라이프오아시스 MAUM 운영 중 간헐적으로 발생하던 요청 실패 장애 대응 기록. 로직상 문제가 없는데도 실패하던 요청의 원인을 Slack 알림 시각과 애플리케이션 로그를 대조해 추적한 끝에, 쓰기 직후 Replica DB를 읽어 발생한 복제 지연(Replication Lag) 문제였음을 확인하고 쓰기 직후 읽기를 Primary DB로 처리해 해결했다."
---

> 라이프오아시스에서 운영하는 서비스 **MAUM**의 **운영 모니터링 및 장애 대응** 기록. 매일 피크 타임 Grafana로 모니터링하고 장애는 Slack 버그 알림봇으로 확인해 대응하는데, 그러던 중 마주친 로직상 문제가 없는데도 간헐적으로 실패하던 요청의 원인을 추적하고 해결한 과정을 정리

## 프로젝트 개요

- **회사 / 서비스**: 라이프오아시스 / MAUM
- **담당 업무**: 운영 모니터링 및 장애 대응 — 복제 지연(Replication Lag)으로 인한 간헐적 요청 실패
- **일정**: 상시 (피크 타임 Grafana 모니터링 담당)
- **기술 스택**: Spring Boot 3 (Kotlin) · MySQL (Primary-Replica) · AbstractRoutingDataSource · Grafana · Slack Bot

**대응 기준**

- 장애가 보고되면 로그 기반으로 원인을 빠르게 확인한다
- 한 번 해결한 장애는 이후 빠르게 대응할 수 있도록 대응 절차로 남긴다

---

## 아키텍처

![MAUM Read/Write DB 분리 구조와 복제 지연 장애 아키텍처 다이어그램. Client 요청이 API 서버(Spring Boot3 Kotlin)로 들어오면, AbstractRoutingDataSource가 @Transactional의 readOnly 여부로 커넥션을 분기한다. readOnly=false인 쓰기 트랜잭션은 Write DB(Primary)로, readOnly=true인 조회 트랜잭션은 Read DB(Replica)로 라우팅된다. Primary의 변경 사항은 비동기 복제로 수백 ms~1초 지연을 두고 Replica에 반영된다. 문제는 쓰기 직후 즉시 같은 데이터를 재조회하는 경우 발생한다 — 그 조회도 readOnly=true로 처리되어 Replica로 라우팅되는데, 복제가 아직 반영되지 않은 시점이면 데이터가 없어 요청이 간헐적으로 실패한다. 각 서버의 에러 로그는 Slack 버그 알림봇으로 실시간 전달되며, 담당자는 알림 시각과 애플리케이션 로그 시각을 대조해 에러 메세지를 특정하고, 복제 지연이 원인임을 확인한 뒤 해당 구간의 읽기를 Write DB로 전환하고, 동일 유형 장애는 복제 지연부터 확인하는 대응 절차로 정립한다](/assets/img/posts/maum-replica-lag-monitoring/maum_replica_lag_architecture.svg)

*MAUM Read/Write DB 분리 구조 — `@Transactional(readOnly)` 여부에 따라 AbstractRoutingDataSource가 조회는 Replica, 쓰기는 Primary로 커넥션을 라우팅한다. Primary → Replica 복제는 비동기라 수백 ms~1초 지연이 생기고, 그 틈에 쓰기 직후 재조회가 들어오면 요청이 실패한다.*

MAUM 백엔드는 **Spring Boot의 `AbstractRoutingDataSource`**로 Read/Write DB를 분리해 두고 있다. `@Transactional(readOnly = true)`로 선언된 조회 트랜잭션은 Read DB(Replica)로, 그 외 쓰기 트랜잭션은 Write DB(Primary)로 커넥션을 라우팅한다. 운영 중에는 매일 피크 타임마다 **Grafana**로 전사 트래픽과 에러율을 모니터링하는 역할을 맡고 있었고, 서버의 에러 로그는 **Slack 버그 알림봇**으로 실시간 전달되어 개별 장애는 이 알림을 보고 대응했다.

---

## 문제

### 1. 로직상 문제없는 요청의 간헐적 실패

Slack 버그 알림봇에 특정 API의 에러 알림이 올라온 걸 확인했다. 코드를 다시 읽어 봐도 로직상 실패할 이유가 없는데, 같은 요청이 어떤 때는 성공하고 어떤 때는 실패했다. 재현도 잘 안 되고 발생 빈도도 낮아서, 코드 리뷰만으로는 원인을 찾을 수 없었다.


---

## 해결

### 1. 알림 시각과 로그 시각을 비교해 에러 메세지 찾기

장애 알림이 온 시각을 기준으로, 이전 구간까지 포함해 애플리케이션 로그를 확인했다.

### 2. 원인 확인 — 쓰기 직후 재조회가 Replica로 라우팅

에러 메세지를 보고 로직에서 실행되는 쿼리가 실제 어떤 DB 를 바라보는지 확인했다. MAUM의 DB 라우팅은 `@Transactional(readOnly)` 값으로 결정된다. 문제가 된 부분은 **쓰기 직후 같은 데이터를 바로 재조회**하는 경우였다.

```
쓰기 트랜잭션 (readOnly=false) → Write DB(Primary) 반영
      ↓ (비동기 복제, 수백 ms~1초 지연)
조회 트랜잭션 (readOnly=true)  → Read DB(Replica) 조회 ← 아직 복제 전이면 데이터 없음
```

재조회도 조회이니 `readOnly = true`로 선언되어 있었고, 그대로 Replica로 라우팅됐다. Primary에 쓴 내용이 Replica에 반영되기까지는 비동기 복제 특성상 수백 ms~1초 정도 지연이 있는데, 그 틈에 재조회가 들어오면 Replica에는 아직 데이터가 없어 요청이 실패했다.

### 3. 쓰기 직후 읽기는 Primary DB로 처리

해당 흐름의 재조회 부분을 쓰기와 같은 트랜잭션(또는 `readOnly = false`)으로 묶어 Write DB를 바로 읽도록 바꿨다. 복제 지연 구간을 건너뛰고 항상 쓰기가 반영된 최신 데이터를 보게 되어, 같은 타이밍 문제가 재발하지 않는다.

### 4. 재발 방지를 위한 대응 절차 정립

같은 유형의 장애가 또 보고되면, 가장 먼저 복제 지연 여부부터 확인하도록 대응 절차에 남겼다.

---

## 결과

- 해당 흐름에서 **동일 유형 장애 재발 0건**.
- 복제 지연을 먼저 확인하는 대응 절차가 생겨 다음번 동일 장애 발생 시 빠르게 대응할 수 있는 워크플로우를 구축했다.

## 회고

처음엔 코드를 몇 번씩 다시 읽어도 멀쩡해 보여서, 간헐적으로만 터지는 이 문제가 더 답답했다. 알림 시각과 로그 시각 사이의 틈을 의심하고 로그를 더 넓게 훑어보면서야 에러 메세지를 잡을 수 있었다. 그 과정에서 Read/Write 분리가 트래픽을 분산시켜 주는게 전부가 아니라는 걸 꺠달았다. — 조회 성능을 위해 Replica로 분산한 대가로, 쓰기 직후 읽기 같은 특정 흐름에서는 복제 지연을 직접 신경 써야 한다는 트레이드오프를 가진 구조였다. 이후로는 새 기능을 설계할 때도 "쓰고 바로 읽는 흐름이 있는가"를 먼저 체크하게 됐다.
