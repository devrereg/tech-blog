---
title: "[라이프오아시스] MAUM Admin 유저 검색 성능 개선 — 800만 유저 LIKE 검색을 Fulltext Index로 전환해 20초에서 200ms로 단축"
date: 2026-09-07 12:00:00 +0900
categories: [포트폴리오]
tags: [mysql, fulltext-index, ngram, query-optimization, spring-boot, kotlin, kubernetes]
description: 라이프오아시스 MAUM Admin의 유저 검색 성능 개선 기록. CS 대응용 유저 조회에서 800만 행을 풀 스캔하느라 20초 이상 걸리던 이름·이메일 LIKE 검색을, ngram 파서 기반 Fulltext Index와 MATCH AGAINST 쿼리로 바꿔 200ms로 줄였다.
---

> 라이프오아시스에서 운영하는 서비스 **MAUM**의 **Admin 유저 검색 성능 개선** 기록이다. CS 담당자가 이름이나 이메일로 유저를 찾을 때 20초 넘게 걸리던 LIKE 검색을 Fulltext Index 기반으로 바꾼 과정을 정리했다.

## 프로젝트 개요

- **회사 / 서비스**: 라이프오아시스 / MAUM
- **프로젝트명**: Admin 유저 검색 성능 개선
- **일정**: 2023.06 (3주)
- **기술 스택**: Spring Boot 3 (Kotlin) · MySQL · Kubernetes (K8s)

**설계 기준**

- 기존에 검색되던 유저가 빠지지 않는 선에서 응답 속도 개선
- 새 인프라를 들이지 않고 MySQL 안에서 해결

---

## 아키텍처

![MAUM Admin 유저 검색 성능 개선 아키텍처 다이어그램](/assets/img/posts/maum-admin-automation/maum_admin_backoffice_architecture_1_user_search.png)

*MAUM Admin 유저 검색 구조 — CS 담당자가 Admin에서 이름·이메일로 유저를 검색하면, API 서버가 약 800만 행의 `user` 테이블을 Fulltext Index로 조회한다.*

전체 작업은 **① 원인 분석 → ② ngram Fulltext Index 생성 → ③ 쿼리 전환 및 검증** 순으로 진행했다.

**1. 원인 분석**

- 실행 계획을 보니 `LIKE '%값%'` 조건 때문에 인덱스를 전혀 쓰지 못하고 `user` 테이블을 처음부터 끝까지 읽고 있었다.

**2. ngram Fulltext Index 생성**

- `name`, `email` 컬럼에 ngram 파서를 쓰는 Fulltext Index를 걸었다.

**3. 쿼리 전환 및 검증**

- 검색 쿼리를 `MATCH ... AGAINST`로 바꾼 뒤, 기존 쿼리와 결과를 맞춰 보고 배포했다.

---

## 문제

### 1. 800만 행 풀 스캔으로 20초 이상 걸리는 유저 검색

CS 담당자는 문의가 오면 이름이나 이메일로 그 유저부터 찾는다. 하루에도 수십 번 쓰는 화면인데, 쿼리는 이렇게 생겼었다.

```sql
SELECT * FROM user
WHERE name LIKE '%값%' OR email LIKE '%값%';
```

앞에 `%`가 붙은 LIKE는 B-Tree 인덱스를 탈 수 없다. 검색할 때마다 800만 행을 전부 읽었고, 유저 한 명 찾는 데 **20초 이상** 걸렸다.

### 2. CS 응대 지연과 DB 부하

문의가 몰리는 시간이면 CS 담당자가 검색 결과를 기다리느라 응대가 줄줄이 밀렸다. 결과가 안 나오니 검색 버튼을 몇 번씩 다시 누르게 되고, 그때마다 풀 스캔 쿼리가 DB에 또 쌓였다.

---

## 해결

### 1. ngram 파서 기반 Fulltext Index 적용

처음엔 Fulltext Index를 기본 파서로 걸었는데, 기본 파서는 공백 기준으로 단어를 자른다. 한글 이름 일부나 이메일 앞부분만 넣으면 검색이 안 됐다.

그래서 문자열을 일정 길이 단위로 잘라 인덱싱하는 ngram 파서로 바꿨다. 이렇게 하면 부분 문자열로 검색해도 인덱스를 탄다.

```sql
CREATE FULLTEXT INDEX idx_name_email
ON user (name, email) WITH PARSER ngram;
```

### 2. MATCH AGAINST 쿼리로 전환

검색 쿼리는 `MATCH ... AGAINST`의 BOOLEAN MODE로 바꿨다. 이제 테이블 전체가 아니라 Fulltext 인덱스에서 후보를 찾는다.

```sql
SELECT * FROM user
WHERE MATCH(name, email) AGAINST(:keyword IN BOOLEAN MODE);
```

### 3. 개선 전후 결과 비교 검증

쿼리 방식이 바뀌면 검색 결과도 달라질 수 있다. CS 담당자들이 실제로 쓰던 검색어를 모아 기존 쿼리와 새 쿼리를 둘 다 돌려 보고, 응답 시간과 결과를 비교했다. 기존에 나오던 유저가 새 쿼리에서도 빠짐없이 나오는 걸 확인하고 나서 적용했다.

---

## 결과

- 유저 검색 응답 시간이 **20초에서 200ms로 줄었다** (약 100배).
- CS 담당자가 검색 결과를 기다리지 않고 바로 응대할 수 있게 됐다.

## 회고

처음에는 Elasticsearch를 붙여야 하나 고민했다. 그런데 Admin 유저 검색 하나 때문에 검색 엔진을 새로 띄우고, MySQL과 데이터 동기화까지 챙기는 건 과하다고 봤다. 우선 MySQL 안에서 할 수 있는 걸 찾아보다가 Fulltext Index를 알게 됐다.

막상 걸어 보니 기본 파서로는 한글 이름이나 이메일 일부가 검색되지 않았고, 그 한계를 해결하려고 찾다가 ngram 파서까지 오게 됐다. 인덱스가 왜 안 타는지, 파서가 문자열을 어떻게 자르는지 하나씩 확인하면서 MySQL 조회 성능을 어떻게 끌어올릴 수 있는지 꽤 깊게 이해하게 된 작업이었다.