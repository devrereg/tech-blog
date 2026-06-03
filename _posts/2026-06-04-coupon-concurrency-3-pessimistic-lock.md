---
title: "쿠폰발급 시스템으로 동시성 제어 구현하기(3) — 비관 락으로 lost update 잡기"
date: 2026-06-04 02:48:23 +0900
categories: [Backend, Concurrency]
tags: [concurrency, coupon, pessimistic-lock, lost-update, kotlin]
description: "락 없이 동시 발급에서 한도를 초과하던 쿠폰발급 서비스를 SELECT … FOR UPDATE(비관 락)로 직렬화해, 재고 100개에 2,000명이 몰려도 정확히 100건만 발급하게 만든다."
---

> 락 없이 한도를 초과하던 발급을 `SELECT … FOR UPDATE`로 직렬화해, 재고 100개에 2,000명이 몰려도 정확히 100건만 발급되게 만들었다. 곁다리로 만난 테스트 오염 이야기는 부록에. (2026-06-04)

## 구현 내용

[지난 글]({% post_url 2026-06-03-coupon-concurrency-2-reproduce-lost-update %})에서, 락이 없는 선착순 쿠폰발급 서비스가 동시 요청에서 한도를 크게 초과 발급함을 가상 스레드 2,000개로 재현했다. 재고 100개짜리 쿠폰에 서로 다른 사용자 2,000명이 동시에 발급을 요청하니 888건이 발급됐다(2편 측정값). 원인은 발급 흐름이 (재고 조회 → 잔여 확인 → 차감)을 락 없이 이어 붙인 read-modify-write라, 수백 개의 트랜잭션이 똑같이 `remainingQuantity = 100`을 읽고 모두 가드를 통과해버리는 것이었다.

이번에는 그 구간을 **비관 락(pessimistic lock)**으로 보호했다. 재고 행을 `SELECT … FOR UPDATE`로 조회해, 같은 행을 건드리는 동시 트랜잭션이 쓰기 락을 두고 줄을 서게 만든다.

핵심은 재고 조회 메서드 하나다. 영속 계층에 비관 쓰기 락 조회를 추가했다.

```kotlin
interface CouponStockJpaRepository : JpaRepository<CouponStock, Long> {
    fun findByCouponId(couponId: Long): CouponStock?

    // SELECT … FOR UPDATE: 해당 재고 row 에 비관적 쓰기 락
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select s from CouponStock s where s.coupon.id = :couponId")
    fun findByCouponIdForUpdate(couponId: Long): CouponStock?
}
```

발급 서비스는 기존 트랜잭션 경계 안에서, 재고 조회만 이 락 조회로 바꿨다.

```kotlin
@Transactional
fun issue(couponId: Long, userId: Long): CouponIssue {
    userRepository.findById(userId) ?: throw UserNotFoundException(userId)

    if (couponIssueRepository.existsByCouponIdAndUserId(couponId, userId)) {
        throw DuplicateIssueException(couponId, userId)
    }

    // 재고 조회 (쿠폰당 1행) — 비관 락으로 동시 차감을 직렬화
    val stock = couponStockRepository.findByCouponIdForUpdate(couponId)
        ?: throw CouponNotFoundException(couponId)

    stock.decrease()

    return couponIssueRepository.save(CouponIssue(couponId = couponId, userId = userId))
}
```

이 한 줄 교체로, 지난 글에서 888건까지 깨지던 동시성 테스트가 통과한다. 성공 정확히 100건, 잔여 0, 발급 이력 100행.

## 구현 내용 설명

### 비관 락이 lost update를 막는 메커니즘

`@Lock(LockModeType.PESSIMISTIC_WRITE)`는 그 조회를 데이터베이스에서 `SELECT … FOR UPDATE`로 변환한다. 이 락은 행 단위 쓰기 락이라, 한 트랜잭션이 재고 행을 잡으면 같은 행을 `FOR UPDATE`로 읽으려는 다른 트랜잭션은 앞 트랜잭션이 커밋(또는 롤백)해 락을 놓을 때까지 그 조회 지점에서 멈춘다.

락이 트랜잭션 경계와 묶이는 게 핵심이다. 발급 서비스는 메서드 전체가 `@Transactional`이고, 락 조회는 그 트랜잭션 안에서 일어난다. 그래서 한 트랜잭션이 (재고 조회 → 차감 → 발급 이력 저장 → 커밋)을 끝낼 때까지 락이 유지되고, 그동안 다른 트랜잭션은 재고 조회에서 대기한다. 결과적으로 read-modify-write 구간 전체가 재고 행 단위로 직렬화된다.

지난 글에서 깨진 이유는 수백 개의 트랜잭션이 차감 직전 똑같은 잔여 수량을 동시에 읽었기 때문이다. 비관 락은 그 "동시에 읽는" 순간 자체를 없앤다. 한 번에 한 트랜잭션만 100을 읽어 99로 줄이고 커밋하면, 다음 트랜잭션이 비로소 99를 읽는다. 100번 차감되어 잔여가 0이 되면, 그 다음 트랜잭션들은 0을 읽고 `decrease()`의 가드에 막혀 `OutOfStockException`으로 실패한다. 그래서 성공은 정확히 100건이다.

### "락"이라는 인프라 관심사를 어디에 둘 것인가

이 프로젝트는 레이어드 아키텍처를 쓰고, 의존 방향은 위에서 아래로 한 방향이다. 도메인 계층은 영속 기술을 모르도록 포트(인터페이스)만 두고, 실제 JPA 구현은 인프라 계층이 제공한다. 그렇다면 `SELECT … FOR UPDATE`라는 명백한 인프라 관심사를 어디에 두어야 도메인의 무지를 깨지 않을까.

택한 방식은 포트에는 "락으로 조회한다"는 의도만 선언하고, 락을 거는 방법은 인프라에만 두는 것이다. 도메인 포트는 이렇게만 안다.

```kotlin
interface CouponStockRepository {
    fun save(stock: CouponStock): CouponStock
    fun findByCouponId(couponId: Long): CouponStock?

    /**
     * 재고 row 를 비관적 쓰기 락(SELECT … FOR UPDATE)으로 조회한다.
     * 동시 발급 시 (조회→차감→커밋)을 row 단위로 직렬화해 lost update 를 막는다.
     * 락은 호출한 트랜잭션이 끝날 때(커밋/롤백) 해제된다.
     */
    fun findByCouponIdForUpdate(couponId: Long): CouponStock?
}
```

`@Lock`이나 `LockModeType` 같은 JPA 타입은 이 인터페이스에 없다. 비관 쓰기 락이 필요하다는 도메인 의도는 메서드 이름과 주석으로 드러나지만, 그걸 `@Lock(PESSIMISTIC_WRITE)`로 실현하는 건 앞서 본 인프라의 `JpaRepository`뿐이다. 도메인은 JPA가 아니라 다른 영속 기술로 갈아끼워도 그대로다.

### 왜 기존 조회 메서드에 락을 붙이지 않았나

이미 `findByCouponId`라는 재고 조회가 있었다. 여기에 `@Lock`을 붙이면 메서드를 새로 만들 필요가 없다. 그런데도 별도 메서드를 추가했다.

조회 메서드는 발급 경로 말고도 쓰일 수 있다. 예를 들어 남은 재고를 단순히 보여주는 읽기 전용 경로에서 같은 조회를 호출할 수 있는데, 거기까지 비관 쓰기 락이 걸리면 불필요하게 다른 트랜잭션을 막는다. 락은 차감하려고 읽는 경로에만 필요하다. 그래서 락이 필요한 발급 경로용으로 `findByCouponIdForUpdate`를 따로 두고, 단순 조회는 락 없는 `findByCouponId`로 남겼다.

### 비관 락이라 재시도가 없다

비관 락은 충돌 자체를 미리 차단한다. 락을 잡지 못한 트랜잭션은 대기하다가, 앞 트랜잭션이 끝나면 갱신된 최신 값을 읽는다. 그래서 "내가 읽은 사이 값이 바뀌었으니 처음부터 다시"라는 재시도 로직이 필요 없다. 그리고 이 경로는 단일 재고 행 하나만 잠근다 — 여러 행을 제각각의 순서로 잠그면 데드락이 생길 수 있지만, 잠그는 행이 하나뿐이라 락 순환 자체가 성립하지 않아 데드락도 없다.

이 점이 다음 단계에서 다룰 낙관 락(optimistic lock)과의 핵심 대비다. 낙관 락은 일단 충돌을 허용하고 커밋 시점에 버전으로 충돌을 감지해 실패시키므로, 호출하는 쪽에 재시도 로직이 반드시 따라붙는다. 비관 락은 대기 비용을, 낙관 락은 재시도 비용을 치른다.

### 줄을 세운다고 선착순은 아니다

한 가지 짚어둘 게 있다. 비관 락은 "한 번에 하나"를 보장하지만, 그게 "먼저 요청한 사람이 먼저 발급받는다"는 선착순까지 보장하진 않는다. 요청이 락에 도달하기까지 스레드 스케줄링, 커넥션 풀 경쟁 같은 단계를 거치면서 도착 순서가 뒤섞이기 때문이다. 즉 락은 초과 발급을 막아 줄 뿐, 정확히 어떤 100명이 그 100장을 가져갈지의 순서까지 맞춰 주지는 않는다. 실제로 이번 동시성 테스트도 "정확히 100명이 성공했는가"만 확인하지 "먼저 온 100명인가"는 보지 않는다.

도착 순서 그대로의 진짜 선착순이 필요하다면, 요청이 들어오는 순간 순번을 매기는 별도 장치(원자적 카운터나 큐)가 따로 있어야 한다 — 다른 글에서 다룬다.

## 다음에 해볼 내용

비관 락으로 정확성은 확보했다. 대신 동시 트랜잭션이 재고 행 하나를 두고 줄을 서므로, 경합이 심하면 대기가 길어진다. 다음 글에서는 다른 접근을 같은 테스트로 비교한다.

- 낙관 락으로 버전 충돌을 감지하고 재시도하는 방법, 그리고 그때 반드시 따라오는 재시도 로직.
- 비관 락(대기 비용)과 낙관 락(재시도 비용)의 트레이드오프를 동시성 테스트의 같은 숫자로 견줘본다.

## 부록 — 회귀 테스트를 빌드에 편입하다 만난 테스트 오염

> 비관 락이라는 본문 주제와는 직접 상관없는 곁다리 기록이다. 깨졌던 동시성 테스트를 이제 통과하니 기본 빌드에 편입했는데, 그 과정에서 락과 무관한 테스트 격리 문제를 만났다. 같은 함정을 피하고 싶은 사람을 위해 따로 남긴다.

락을 넣어 동시성 테스트를 단독으로 돌리면 깨끗하게 통과한다. 그래서 이 테스트를 기본 빌드에 편입했다. 지난 글에서는 이 테스트에 `concurrency` 태그를 붙여 기본 빌드에서 제외했었는데(아직 락이 없어 항상 실패했으니까), 이제 통과하니 태그를 떼고 평범한 회귀 테스트로 만들었다.

구체적으로는 2편의 테스트 설정을 들어냈다. 태그 필터(`excludeTags("concurrency")`)와, 그 태그만 따로 돌리던 `concurrencyTest` 태스크를 둘 다 지우고, 모든 테스트를 한 번에 돌리는 기본 설정으로 단순화했다.

```kotlin
// 2편의 excludeTags("concurrency") + 별도 concurrencyTest 태스크를 제거.
// 이제 기본 빌드가 동시성 테스트까지 함께 돌린다.
tasks.withType<Test> {
    useJUnitPlatform()
}
```

그런데 전체 빌드를 돌리자 동시성 테스트가 발급행 101로 깨졌다. 이 테스트는 그동안 늘 혼자 떼어 돌렸는데, 빌드에 편입하면서 처음으로 다른 테스트들과 한 묶음으로 함께 돌게 됐다. 그러자 혼자 돌리면 100, 다른 테스트들과 함께 돌리면 101. 게다가 매번이 아니라 실행 순서에 따라 그때그때 달라지는, 간헐적으로 깨지는(flaky) 실패였다.

원인은 락이 아니었다. **테스트 간 데이터 오염**이었다.

발급 REST API를 검증하는 다른 통합 테스트(`CouponIssueControllerTest`)가 있는데, 이 테스트도 실제 데이터베이스를 쓰는 통합 테스트(`@SpringBootTest`)였다. 발급 성공 케이스를 검증하면서 실제 발급 이력 행을 커밋하고, 지우지 않았다. 한편 동시성 테스트는 발급 행 수를 전역 `count()`로 세고 있었다. 그래서 발급 API 테스트가 먼저 돌아 행을 하나 남기면, 뒤이은 동시성 테스트의 전역 `count()`가 그 잔여 행까지 합산해 101이 됐다. 순서가 반대면 100. 전형적인 순서 의존 flaky다.

여기서 배운 것 셋.

**첫째, `@SpringBootTest`는 롤백하지 않는다.** 슬라이스 테스트나 `@Transactional`을 단 테스트는 끝에서 자동 롤백되지만, 그냥 `@SpringBootTest`만 단 통합 테스트는 그렇지 않다. 발급 API 테스트는 커밋한 데이터를 직접 정리하거나, 테스트를 트랜잭션으로 감싸 격리해야 했다. 지난 글에서 동시성 테스트가 `@AfterEach`로 데이터를 손수 지운 것과 같은 이유다 — 다만 그 교훈을 다른 테스트에는 적용하지 않았던 게 이번 함정이었다.

**둘째, 전역 `count()` 단언은 다른 테스트에 취약하다.** 동시성 테스트는 "이 쿠폰이 정확히 100번 발급됐는가"를 묻고 싶었던 것이지, "데이터베이스 전체에 발급 행이 100개인가"를 물은 게 아니었다. 전역 `count()`는 후자라, 다른 테스트가 남긴 행에 그대로 오염된다. 이 쿠폰의 발급 수만 세는 단언으로 바꾸면 다른 테스트가 무엇을 남기든 영향받지 않는다.

```kotlin
// 전역 count() → 이 쿠폰의 발급 수만 (순서 독립)
val issued = couponIssueJpaRepository.countByCouponId(couponId)
// ... (중간 단언 생략)
assertThat(issued).isEqualTo(TOTAL_QUANTITY.toLong())
```

**셋째, "혼자 돌릴 때 green"이 "다른 테스트들과 함께 돌릴 때 green"을 보장하지 않는다.** 한 테스트만 떼어 돌리면 깨끗한 상태에서 시작하므로 다른 테스트가 남긴 오염을 못 본다. 테스트의 정확성은 자기 자신뿐 아니라 함께 도는 이웃에도 달려 있다.

고친 방법은 두 가지였다. 동시성 테스트의 단언을 전역 `count()`에서 이 쿠폰의 발급 수만 세는 쪽으로 바꿔 순서 독립으로 만들었고, 발급 API 테스트에는 `@Transactional`을 달아 커밋 자체가 새지 않게 막았다.

```kotlin
@SpringBootTest
@AutoConfigureMockMvc
@Transactional
class CouponIssueControllerTest(...)
```

여기서 한 가지가 의외였다. 발급 API 테스트는 MockMvc로 컨트롤러를 호출하고, 컨트롤러는 다시 발급 서비스의 `@Transactional` 메서드를 부른다. 테스트에 `@Transactional`을 달면 테스트가 트랜잭션을 하나 열고, 그 안에서 호출된 서비스의 `@Transactional`은 기본 전파 방식(REQUIRED)에 따라 새 트랜잭션을 열지 않고 테스트가 연 트랜잭션에 합류한다. MockMvc 호출이 같은 스레드에서 동기로 일어나기에 가능한 일이다. 그래서 테스트가 끝날 때 바깥 트랜잭션이 통째로 롤백되면 서비스가 만든 발급 행도 함께 사라진다 — 커밋 누수가 막힌다.

다만 이 격리는 같은 스레드에서 트랜잭션을 공유할 수 있을 때만 통한다. 동시성 테스트처럼 여러 스레드가 각자 독립 트랜잭션을 커밋해야 의미가 있는 테스트에는 `@Transactional`을 쓸 수 없고(그러면 경쟁할 트랜잭션 경계가 사라진다), 지난 글처럼 `@AfterEach`로 손수 정리해야 한다. 같은 통합 테스트라도 격리 전략이 다르다는 점이 이번에 또렷해졌다.
