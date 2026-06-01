---
title: "Flyway란 무엇인가? 데이터베이스 변경 이력 관리와 실무 운영"
date: 2026-06-01 14:00:00 +0900
categories: [Backend, Database]
tags: [flyway, migration, jpa, hibernate, spring-boot]
description: "Flyway는 무엇이고 ddl-auto와 어떻게 다른가? 마이그레이션 파일 관리부터 운영 환경에서의 사용법, Liquibase와의 비교까지 정리한다."
---

## 들어가며

Spring Boot 프로젝트를 진행하다 보면 다음과 같은 설정을 자주 볼 수 있다.

```yaml
spring:
  flyway:
    enabled: true

  jpa:
    hibernate:
      ddl-auto: validate
```

또한 프로젝트 구조에는 다음과 같은 파일이 존재한다.

```text
db/migration
  V1__init.sql
  V2__add_email.sql
  V3__create_order.sql
```

처음 접했을 때는 다음과 같은 의문이 생길 수 있다.

- Flyway는 무엇인가?
- `ddl-auto=update`와 어떤 차이가 있는가?
- SQL 파일을 직접 작성해야 하는가?
- SQL은 누가 실행하는가?
- 여러 개발자가 동시에 작업하면 버전 충돌은 어떻게 해결하는가?

이 글에서는 Flyway의 개념과 동작 방식, 그리고 실무에서 사용되는 이유를 정리한다.

## Flyway란?

Flyway는 **데이터베이스 스키마 변경 이력을 관리하는 마이그레이션(Migration) 도구**이다.

애플리케이션이 성장하면서 데이터베이스 구조는 지속적으로 변경된다. 예를 들어 다음과 같은 변화가 발생할 수 있다.

```text
users 테이블 생성
        ↓
  email 컬럼 추가
        ↓
  phone 컬럼 추가
        ↓
 orders 테이블 생성
```

Flyway는 이러한 변경 이력을 파일 단위로 관리한다.

```text
V1__create_users.sql
V2__add_email.sql
V3__add_phone.sql
V4__create_orders.sql
```

각 파일은 데이터베이스 변경 작업을 의미하며, Flyway는 이를 순서대로 실행한다.

## Flyway는 Spring Boot 전용 도구인가?

아니다. Flyway는 **자바 기반의 독립 마이그레이션 도구**다. 실행 방식은 다음과 같이 다양하다.

- Maven / Gradle 플러그인
- CLI (Flyway 단독 실행)
- Docker 이미지
- JVM 애플리케이션 내부 통합

그리고 다음과 같은 JVM 프레임워크와 통합되어 자주 쓰인다.

- Spring Boot
- Spring Framework
- Quarkus
- Micronaut

이 글은 Spring Boot 기준으로 설명하지만, Flyway 자체는 Spring Boot에 묶여 있는 도구가 아니다. 다만 Spring Boot는 Flyway와의 통합을 기본적으로 지원하기 때문에 사용이 편리한 편이다.

## ddl-auto=update 방식

Flyway를 도입하기 전, 비교 대상으로 자주 등장하는 것이 Hibernate의 `ddl-auto=update` 설정이다. Spring Boot에서는 다음과 같이 설정할 수 있다.

```yaml
spring:
  jpa:
    hibernate:
      ddl-auto: update
```

예를 들어 엔티티가 다음과 같다고 가정한다.

```java
@Entity
public class User {

    @Id
    private Long id;

    private String name;
}
```

애플리케이션이 실행되면 Hibernate가 자동으로 테이블을 생성한다.

```sql
CREATE TABLE users (
    id BIGINT NOT NULL,
    name VARCHAR(255),
    PRIMARY KEY (id)
);
```

이후 엔티티에 컬럼을 추가하면 데이터베이스도 자동으로 수정된다.

```java
private String email;
```

↓

```sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
```

## Flyway 방식

Flyway를 사용하면 데이터베이스 변경을 SQL 파일로 관리한다.

예를 들어 `users` 테이블을 생성하려면 다음 파일을 작성한다.

**V1__create_users.sql**

```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255)
);
```

이후 `email` 컬럼을 추가하려면 새로운 파일을 생성한다.

**V2__add_email.sql**

```sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
```

`phone` 컬럼을 추가할 때도 마찬가지다.

**V3__add_phone.sql**

```sql
ALTER TABLE users ADD COLUMN phone VARCHAR(30);
```

Flyway는 파일을 순서대로 실행하여 데이터베이스 상태를 관리한다.

## SQL 파일은 직접 작성해야 하는가?

그렇다. Flyway는 SQL을 자동으로 생성하지 않는다.

다음과 같은 변경이 필요하다면

```java
private String email;
```

개발자가 직접 SQL을 작성해야 한다.

```sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
```

**Flyway의 목적은 자동 생성이 아니라 변경 이력 관리에 있다.**

## SQL은 직접 실행해야 하는가?

아니다. SQL 파일은 직접 작성하지만 **실행은 Flyway가 수행한다**.

애플리케이션 시작 시 Flyway는 다음 과정을 수행한다.

1. 현재 DB 버전 확인
2. 실행되지 않은 SQL 파일 검색
3. SQL 실행
4. 실행 이력 저장

개발자가 데이터베이스 콘솔에 접속하여 직접 SQL을 실행할 필요는 없다.

## Flyway는 실행 이력을 어떻게 관리하는가?

Flyway는 내부적으로 관리 테이블을 생성한다.

```text
flyway_schema_history
```

실제 테이블에는 다음과 같은 컬럼이 포함된다.

| version | description  | checksum    | installed_on        | success |
|---------|--------------|-------------|---------------------|---------|
| 1       | create_users | 1284736924  | 2026-06-01 14:00:00 | true    |
| 2       | add_email    | 2384957123  | 2026-06-02 09:30:00 | true    |
| 3       | add_phone    | 9237461529  | 2026-06-02 15:20:00 | true    |

핵심 컬럼의 역할은 다음과 같다.

- `version` — 마이그레이션 버전
- `description` — 마이그레이션 설명
- `checksum` — SQL 파일의 체크섬. 이미 실행된 파일이 변경되지 않았는지 검증할 때 사용
- `installed_on` — 실행 시각
- `success` — 실행 성공 여부

Flyway는 이 정보를 기반으로 어떤 SQL이 이미 실행되었는지 판단하고, 파일이 변경됐는지도 함께 확인한다.

## ddl-auto=validate의 역할

실무에서는 다음과 같은 설정을 많이 사용한다.

```yaml
spring:
  flyway:
    enabled: true

  jpa:
    hibernate:
      ddl-auto: validate
```

여기서 Flyway와 JPA의 역할은 분리된다.

```text
Flyway       → 테이블 생성 및 변경
JPA validate → Entity와 DB 구조 검증
```

예를 들어 Entity에는 `email` 컬럼이 존재하지만

```java
private String email;
```

DB에는 `email` 컬럼이 없다면

```text
Schema-validation: missing column [email]
```

오류가 발생하며 애플리케이션이 시작되지 않는다.

참고로 `validate`는 테이블/컬럼 존재 여부와 컬럼 타입·길이·NOT NULL 같은 기본 제약을 검증한다. 다만 인덱스, 외래 키, 컬럼의 기본값 같은 항목은 검증하지 않는다.

## 이미 실행된 Migration 파일은 수정하면 안 된다

예를 들어 다음 파일이 운영 환경에 적용되었다고 가정한다.

**V1__create_users.sql**

```sql
CREATE TABLE users (
    id BIGINT
);
```

이후 파일을 수정하면 문제가 발생한다.

```sql
CREATE TABLE users (
    id BIGINT,
    name VARCHAR(255)
);
```

Flyway는 실행 당시의 SQL과 현재 SQL이 다르다는 사실을 감지한다(체크섬 검증).

실무에서는 기존 Migration 파일을 수정하지 않고, 대신 새로운 Migration 파일을 추가한다.

**V2__add_name.sql**

```sql
ALTER TABLE users ADD COLUMN name VARCHAR(255);
```

## Flyway를 Git과 비교해보기

Flyway는 Git과 유사한 개념으로 이해할 수 있다.

Git

```text
Commit 1
Commit 2
Commit 3
```

Flyway

```text
V1
V2
V3
```

Git에서 과거 Commit을 수정하지 않듯이, Flyway도 과거 Migration을 수정하지 않는다. 새로운 버전을 추가하여 변경 이력을 관리한다.

## 운영 환경에서 DB를 직접 수정하면 안 되는 이유

운영 환경에서 다음 SQL을 직접 실행했다고 가정한다.

```sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
```

이 경우 데이터베이스 상태와 Flyway 이력이 일치하지 않게 된다.

```text
Flyway 이력 ≠ 실제 DB 상태
```

결과적으로 개발 환경, 운영 환경, 테스트 환경의 상태가 달라질 수 있다. 데이터베이스 변경은 반드시 Flyway를 통해 관리하는 것이 일반적이다.

## 데이터 마이그레이션도 가능하다

Flyway는 단순한 테이블 생성 도구가 아니다. **데이터 변경 작업도 수행할 수 있다.**

예를 들어 `nickname` 컬럼을 `username`으로 변경하는 경우

```sql
ALTER TABLE users ADD COLUMN username VARCHAR(255);

UPDATE users SET username = nickname;

ALTER TABLE users DROP COLUMN nickname;
```

처럼 데이터 이동 작업까지 한 마이그레이션 파일 안에 포함할 수 있다.

> 다만 MySQL처럼 **DDL이 암묵적 커밋(implicit commit)을 발생시키는** DB에서는 위처럼 DDL과 DML을 한 파일에 섞으면 중간에 실패해도 부분적으로 적용된 상태가 남는다. PostgreSQL은 트랜잭션 안에서 DDL을 처리할 수 있어 상대적으로 안전하지만, 안전한 운영을 위해 컬럼 추가 → 데이터 이동 → 컬럼 제거를 **여러 마이그레이션 파일과 배포 주기로 분리**하는 패턴도 자주 쓰인다.

## 여러 개발자가 작업하면 버전 충돌은 어떻게 해결할까?

다음과 같은 상황을 생각해 볼 수 있다.

개발자 A

```text
V2__add_email.sql
```

개발자 B

```text
V2__add_phone.sql
```

동시에 생성했다면 버전 충돌이 발생한다.

이를 해결하기 위해 버전 번호를 **시간 기반(Timestamp)** 으로 관리하는 경우가 많다.

```text
V202606011000__create_users.sql
V202606011030__add_email.sql
V202606011100__add_phone.sql
```

Timestamp 기반 버전 전략은 충돌 가능성을 크게 줄여 준다. 다만 이는 Flyway가 강제하는 기능이 아니라 팀에서 정해서 쓰는 **파일명 컨벤션**이다. Flyway는 `V`로 시작하는 버전(versioned) 마이그레이션과 `R`로 시작하는 반복(repeatable) 마이그레이션을 구분할 뿐, 버전 번호 자체는 정수든 타임스탬프든 자유롭게 쓸 수 있다.

## Flyway와 Liquibase

Flyway와 자주 비교되는 도구로 Liquibase가 있다.

### Flyway

- SQL 중심
- 설정이 단순함
- 학습 비용이 낮음

```sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
```

### Liquibase

- 변경 관리 기능이 풍부함
- 오픈소스 기준으로 롤백(rollback) 정의가 풍부함 — Flyway는 유료(Teams) 에디션에서 undo 마이그레이션을 지원
- 설정이 복잡함

```xml
<changeSet id="1" author="admin">
    <addColumn tableName="users">
        <column name="email" type="varchar(255)"/>
    </addColumn>
</changeSet>
```

Spring Boot 프로젝트에서는 Flyway가 더 널리 사용되는 편이다.

## 일반적인 프로젝트 구조

```text
src/main/resources
└── db
    └── migration
        ├── V202606011000__create_users.sql
        ├── V202606011030__add_email.sql
        ├── V202606011100__add_phone.sql
        └── V202606011200__create_orders.sql
```

## 정리

Flyway는 데이터베이스 변경 이력을 관리하기 위한 도구이다. 핵심을 정리하면 다음과 같다.

1. Flyway는 데이터베이스 버전 관리 도구이다.
2. SQL 파일은 직접 작성하지만 실행은 Flyway가 수행한다.
3. 이미 적용된 Migration 파일은 수정하지 않는다.
4. 새로운 변경은 새로운 Migration 파일로 관리한다.
5. 운영 환경의 데이터베이스 변경도 Flyway를 통해 관리한다.
6. 실무에서는 `Flyway + ddl-auto=validate` 조합이 널리 사용된다.
7. Timestamp 기반 버전 전략을 사용하면 충돌을 줄일 수 있다.

소스코드 변경 이력을 Git으로 관리하듯이, 데이터베이스 변경 이력은 Flyway로 관리할 수 있다. 이는 환경 간 일관성을 유지하고 데이터베이스 변경 내역을 추적하는 데 중요한 역할을 한다.
