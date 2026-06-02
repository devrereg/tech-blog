---
title: "쿠폰발급 시스템으로 동시성 제어 구현하기(1) — 락 없는 쿠폰 발급 1차 구현과 그 한계"
date: 2026-06-02 01:37:56 +0900
categories: [Backend, Concurrency]
tags: [concurrency, coupon, kotlin, spring-boot, lost-update]
description: "쿠폰 선착순 발급 도메인을 락 없이 1차 구현하고, lost update와 1인 1매 이중 방어가 필요한 이유를 정리한다."
---

> 선착순 쿠폰 발급 도메인을 레이어드 아키텍처로 구현했다. 발급의 핵심 흐름을 먼저 세운, 동시성 제어는 아직 없는 1차 버전이다. (2026-06-02)

## 구현 내용

선착순 쿠폰 발급 기능을 Spring boot3와 kotlin + 레이어드 아키텍처로 구현했다. 엔티티 4종(`User`, `Coupon`, `CouponStock`, `CouponIssue`), 리포지토리, 그리고 발급 유스케이스인 `CouponIssueService`까지. 단일 스레드 테스트는 통과한다.

이번 1차 구현에는 동시성 제어(락, 원자적 UPDATE, 분산 락)가 아직 없다. 발급 도메인과 흐름을 정확히 세우는 데 먼저 집중했다.

발급 서비스의 전부는 이 한 메서드다

```kotlin
@Transactional
fun issue(couponId: Long, userId: Long): CouponIssue {
    // 사용자 존재 확인
    userRepository.findById(userId) ?: throw UserNotFoundException(userId)

    // 1인 1매: 이미 발급받았으면 중복(DB unique 제약과 이중 방어)
    if (couponIssueRepository.existsByCouponIdAndUserId(couponId, userId)) {
        throw DuplicateIssueException(couponId, userId)
    }

    // 재고 조회 (쿠폰당 1행)
    val stock = couponStockRepository.findByCouponId(couponId)
        ?: throw CouponNotFoundException(couponId)

    // 재고 차감
    stock.decrease()

    // 발급 이력 저장
    return couponIssueRepository.save(CouponIssue(couponId = couponId, userId = userId))
}
```

재고 차감 로직은 도메인 엔티티가 소유한다.

```kotlin
fun decrease() {
    if (remainingQuantity <= 0) {
        throw OutOfStockException()
    }
    remainingQuantity -= 1
}
```

## 구현 내용 설명

### 발급 흐름과 책임 분리

`issue()`의 흐름은 단순하다. **사용자 확인 → 중복 확인 → 재고 조회 → 차감 → 이력 저장**이고, 트랜잭션 경계는 이 메서드(`@Transactional`)다. 서비스는 오케스트레이션만 하고, "재고를 어떻게 줄이는가"라는 규칙은 도메인 엔티티(`CouponStock.decrease()`)가 가진다. `decrease()`는 0 미만 차감을 막고 소진 시 `OutOfStockException`을 던진다 — 그 자체로는 정확하고, 단일 스레드 테스트도 통과한다.

### 재고를 별도 엔티티로 분리한 이유

재고를 `Coupon`의 컬럼이 아니라 **별도 `CouponStock` 엔티티(쿠폰당 1행)**로 분리했다. 지금만 보면 굳이 나눌 이유가 없어 보이지만, 재고 행 하나가 곧 동시 접근이 몰리는 지점이다. 잔여 수량을 독립된 행으로 두면 이후 그 행을 단위로 락을 걸거나 원자적으로 갱신하기가 쉬워진다.

## 구현의 한계 / 문제점

이 코드는 단일 스레드에서는 정확하지만, 동시 요청이 몰리면 무너진다. 그리고 깨지는 모양이 직관과 다르다.

### 1. read-modify-write — lost update

`decrease()`는 사실상 **read-modify-write**(읽고 → 계산하고 → 쓰기) 패턴이다. "read"는 트랜잭션 안에서 조회한 `stock.remainingQuantity`이고, "write"는 커밋 시점의 UPDATE다.

락이 없으면 두 트랜잭션 T1, T2가 같은 `remainingQuantity = 100`을 각자 읽고, 각자 99로 계산해 쓴다. 둘 다 커밋하면 결과는 99 — 2개가 나갔는데 1개만 줄어든 **lost update(갱신 분실)**다. `decrease()`가 단일 스레드에서 옳다고 동시성에서 옳은 게 아니다.

### 2. 재고 음수는 DB가 막는다 → 깨짐은 "발급 수 > 한도"로 샌다

락이 없으면 흔히 "재고가 음수로 내려간다"를 떠올린다. 그런데 이 스키마에서는 그렇게 안 될 가능성이 크다. 마이그레이션에 CHECK 제약이 있고

```sql
CONSTRAINT ck_coupon_stocks_remaining_non_negative CHECK (remaining_quantity >= 0)
```

엔티티 생성 시점의 `require(remainingQuantity >= 0)` 검증과 `decrease()`의 `<= 0` 가드까지 더하면, **잔여 수량이 음수가 되는 일은 사실상 막혀 있다.**

대신 lost update의 본질은 "차감이 덜 됐다"이다. 100개 재고에 동시 요청이 몰리면 `remainingQuantity`는 0 근처에서 멈추는데, **발급 이력(`coupon_issues`) 건수는 100을 초과**할 수 있다. 즉 정합성 깨짐은 "음수 재고"가 아니라 **`COUNT(coupon_issues) > coupon.totalQuantity`** 형태로 나타날 거라는 예측이다.

### 3. 1인 1매도 같은 함정 — 그래서 이중 방어

`existsByCouponIdAndUserId`로 중복을 막지만, 이 "확인 후 저장"도 read-modify-write다. 같은 사용자의 요청 두 개가 동시에 들어오면 둘 다 `exists == false`를 읽고 둘 다 저장으로 진행할 수 있다 — 체크를 통과해버린다.

그래서 최후 방어선은 DB 제약이다:

```sql
CONSTRAINT uq_coupon_issues_coupon_user UNIQUE (coupon_id, user_id)
```

애플리케이션 체크를 race가 뚫더라도, 두 번째 INSERT는 unique 제약 위반으로 `DataIntegrityViolationException`을 맞고 롤백된다. 핵심은 **"애플리케이션 체크 + DB 제약" 이중 방어가 왜 둘 다 필요한가**이다.

- **애플리케이션 체크**: 정상 경로에서 친절한 도메인 예외(`DuplicateIssueException`)를 주고, 불필요한 INSERT를 줄인다.
- **DB unique 제약**: 동시성 race를 막는 진짜 보증. 단일 행 INSERT의 원자성과 제약 검사는 DB가 보장하므로 애플리케이션 레벨 race와 무관하게 중복을 거른다.

체크만 있으면 동시성에서 뚫리고, 제약만 있으면 정상 경로에서도 거친 예외가 나온다. 그래서 둘 다 둔다.

## 다음에 해볼 내용

다음 글에서는 재고 100개를 2,000명이 동시에 발급하는 다중 스레드 테스트를 작성해, 위에서 짚은 한계가 실제로 어떻게 드러나는지 확인한다.

- 발급 이력 건수가 `totalQuantity`를 초과하는지(lost update) 측정한다.
- 같은 사용자 동시 요청에서 unique 제약이 실제로 중복을 거르는지 확인한다.
- 그 깨짐을 확인한 뒤, 비관 락 / 낙관 락 / 원자적 UPDATE를 차례로 적용하며 해결 과정과 트레이드오프를 비교한다.
