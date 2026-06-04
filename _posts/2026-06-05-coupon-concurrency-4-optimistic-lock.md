---
title: "쿠폰발급 시스템으로 동시성 제어 구현하기(4) — 낙관 락 발급 경로와 비관 락의 재시도 비용 비교"
date: 2026-06-05 01:07:05 +0900
categories: [Backend, Concurrency]
tags: [concurrency, coupon, optimistic-lock, retry, kotlin]
description: "비관 락으로 막아 둔 선착순 발급에 낙관 락(@Version) + 수동 재시도 경로를 나란히 만들어 같은 동시성 테스트로 견줘본다. 두 경로 모두 정확히 100건만 발급하지만, 낙관 락은 그 정확성을 재시도 비용으로 산다."
---

> 비관 락으로 막아 둔 선착순 발급에, 낙관 락(`@Version`) + 수동 재시도라는 두 번째 경로를 나란히 만들어 같은 동시성 테스트로 견줘봤다. 재고 100개에 2,000명이 몰리는 동일 조건에서 두 경로 모두 정확히 100건만 발급하지만, 낙관 락은 그 정확성을 "재시도"라는 비용으로 산다. (2026-06-05)

## 구현 내용

[지난 글]({% post_url 2026-06-04-coupon-concurrency-3-pessimistic-lock %})에서 선착순 쿠폰발급을 비관 락(`SELECT … FOR UPDATE`)으로 직렬화해, 재고 100개에 2,000명이 동시에 몰려도 정확히 100건만 발급되게 만들었다. 비관 락은 같은 재고 행을 건드리는 트랜잭션을 줄 세워 "동시에 읽는" 순간 자체를 없앤다.

이번에는 같은 문제를 다른 방식으로 푸는 두 번째 발급 경로를 추가했다. **낙관 락(optimistic lock)**이다. 충돌을 미리 막는 대신 일단 허용하고, 커밋 시점에 행 버전으로 충돌을 감지해 실패시킨 뒤, 호출하는 쪽에서 다시 시도한다.

먼저 재고 엔티티에 버전 컬럼을 두었다. JPA에서는 `@Version` 필드 하나로 충분하다.

```kotlin
@Entity
@Table(name = "coupon_stocks")
class CouponStock(
    // ... coupon, remainingQuantity, id
) {
    /**
     * 낙관 락(@Version)용 버전. 동시 갱신 충돌 시 OptimisticLockingFailureException 으로 드러난다.
     * 생성자로 노출하지 않고 0 으로 시작하며, 갱신마다 Hibernate 가 증가시킨다.
     */
    @Version
    @Column(name = "version", nullable = false)
    var version: Long = 0

    // ...
}
```

데이터베이스 쪽에도 컬럼을 추가하는 마이그레이션을 더했다.

```sql
ALTER TABLE coupon_stocks ADD COLUMN version BIGINT NOT NULL DEFAULT 0;
```

발급 "한 번의 시도"는 비관 경로와 거의 같다. 단 하나, 재고를 락 없이 읽는다는 점만 다르다.

```kotlin
@Service
class OptimisticCouponIssueAttempt(
    private val userRepository: UserRepository,
    private val couponStockRepository: CouponStockRepository,
    private val couponIssueRepository: CouponIssueRepository,
) {

    @Transactional
    fun attempt(couponId: Long, userId: Long): CouponIssue {
        userRepository.findById(userId) ?: throw UserNotFoundException(userId)

        if (couponIssueRepository.existsByCouponIdAndUserId(couponId, userId)) {
            throw DuplicateIssueException(couponId, userId)
        }

        // 재고 조회 (쿠폰당 1행) — 락 없음. 동시성은 @Version 낙관 락 + 바깥 재시도로 보장한다.
        val stock = couponStockRepository.findByCouponId(couponId)
            ?: throw CouponNotFoundException(couponId)

        stock.decrease()

        return couponIssueRepository.save(CouponIssue(couponId = couponId, userId = userId))
    }
}
```

그리고 이 시도를 감싸는 재시도 루프를 별도 빈으로 두었다. 이 빈에는 `@Transactional`이 없다.

```kotlin
@Service
class OptimisticCouponIssueService(
    private val attempt: OptimisticCouponIssueAttempt,
) {

    fun issue(couponId: Long, userId: Long): OptimisticIssueResult {
        var lastConflict: OptimisticLockingFailureException? = null

        for (attemptNo in 1..MAX_ATTEMPTS) {
            try {
                val issue = attempt.attempt(couponId, userId)
                return OptimisticIssueResult(issue = issue, attempts = attemptNo)
            } catch (e: OptimisticLockingFailureException) {
                // 충돌만 재시도 — 다음 시도는 새 트랜잭션에서 최신 버전을 다시 읽는다.
                lastConflict = e
            }
        }

        throw lastConflict!!
    }

    companion object {
        const val MAX_ATTEMPTS = 100
    }
}
```

같은 동시성 테스트(재고 100, 서로 다른 사용자 2,000명, 가상 스레드로 동시 발사)를 이 경로에 통과시키면, 비관 경로와 똑같은 불변식이 성립한다. 성공 정확히 100건, 잔여 0, 발급 이력 100행.

## 구현 내용 설명

### 낙관 락이 충돌을 감지하는 메커니즘

`@Version` 필드가 있으면 Hibernate는 그 엔티티를 갱신할 때 버전을 `WHERE` 조건에 넣는다. 즉 차감 후 커밋(flush) 시점에 나가는 SQL이 이런 모양이 된다.

```sql
UPDATE coupon_stocks
   SET remaining_quantity = ?, version = ?    -- version = 읽었던 값 + 1
 WHERE id = ? AND version = ?                  -- version = 읽었던 값
```

핵심은 `WHERE` 절의 `version = 읽었던 값`이다. 두 트랜잭션이 락 없이 같은 재고 행을 읽으면 둘 다 같은 `version`을 손에 쥔다. 한쪽이 먼저 커밋해 버전을 올리면, 뒤따라 커밋하려는 쪽의 `UPDATE`는 `WHERE version = 옛 값`에 걸리는 행이 더 이상 없어 영향 행이 0이 된다. Hibernate는 이 0을 "내가 읽은 사이 누가 바꿨다"는 신호로 읽어 `StaleObjectStateException`을 던지고, Spring이 이를 `OptimisticLockingFailureException`으로 번역한다.

비관 락이 "동시에 읽는 순간"을 없앤다면, 낙관 락은 그 순간을 허용하되 커밋할 때 들통나게 한다. 충돌을 막는 게 아니라 사후에 감지하는 방식이라, 감지된 충돌을 어떻게 회복할지가 곧바로 따라온다. 그게 재시도다.

### 재시도는 왜 트랜잭션 바깥이어야 하나

낙관 충돌은 flush/커밋 시점에 터진다. 그리고 그 예외가 던져지는 순간, 그 트랜잭션은 이미 rollback-only로 표시되어 더 쓸 수 없는 상태가 된다. 영속성 컨텍스트(1차 캐시)도 충돌을 일으킨 옛 버전의 엔티티를 그대로 쥔 채 죽는다.

그래서 같은 트랜잭션 안에서 다시 `decrease()`를 부르는 식의 재시도는 의미가 없다. 트랜잭션은 이미 못 쓰는 상태이고, 캐시에는 여전히 옛 버전이 남아 있어 다시 시도해도 같은 충돌이 날 뿐이다. 재시도가 성립하려면 매 시도가 새 트랜잭션·새 영속성 컨텍스트에서 최신 버전을 다시 읽어야 한다.

이 제약이 빈을 둘로 나눈 이유다. 한 번의 시도(`attempt`)는 `@Transactional`을 달아 자기만의 트랜잭션 경계를 갖고, 재시도 루프(`issue`)는 트랜잭션 바깥에 있어야 한다. 루프가 `for` 안에서 `attempt`를 부를 때마다 트랜잭션이 새로 열리고 닫히므로, 매 시도는 깨끗한 영속성 컨텍스트에서 갱신된 `version`을 다시 읽는다.

### 자기호출(self-invocation) 함정 때문에 빈을 분리했다

여기서 한 가지 함정을 피해야 했다. Spring의 `@Transactional`은 AOP(Aspect-Oriented Programming) 프록시로 동작한다. 빈을 호출하면 실제 객체가 아니라 그 객체를 감싼 프록시가 먼저 트랜잭션을 열고, 끝나면 커밋·롤백을 처리한다.

그런데 같은 빈 안에서 `this.attempt()`를 부르면 프록시를 거치지 않고 내부 메서드를 직접 호출한다. 그러면 `@Transactional`이 통째로 무시된다. 재시도 루프와 단일 시도를 한 빈 안 두 메서드로 두고 루프가 `attempt()`를 부르면, 매 시도가 새 트랜잭션을 여는 게 아니라 트랜잭션 없이 돌아 의도가 깨진다.

그래서 단일 시도(`OptimisticCouponIssueAttempt`, `@Transactional`)와 재시도 루프(`OptimisticCouponIssueService`, 트랜잭션 없음)를 별도 빈으로 분리하고, 루프가 시도를 생성자 주입으로 받아 호출하게 했다. 이러면 호출이 프록시를 경유하므로 매 시도마다 `@Transactional`이 정상 적용된다.

### 무엇을 재시도하고 무엇을 재시도하지 않을 것인가

재시도 루프는 `OptimisticLockingFailureException`만 잡는다. 이게 정확성의 핵심이다.

발급 시도가 던지는 예외에는 두 부류가 있다. 하나는 낙관 충돌 — "내가 읽은 사이 누가 바꿨다"는, 다시 읽으면 풀릴 수 있는 일시적 실패다. 다른 하나는 재고 소진(`OutOfStockException`), 중복 발급(`DuplicateIssueException`), 사용자 없음(`UserNotFoundException`) 같은 정당한 비즈니스 실패다. 후자는 다시 시도해도 결과가 바뀌지 않는다. 재고가 0이면 백 번을 다시 읽어도 0이다.

만약 모든 예외를 싸잡아 재시도하면, 재고가 소진된 뒤 들어온 1,900명이 각자 `OutOfStockException`을 받고도 100번씩 재시도하며 헛돌게 된다. 재시도 대상을 충돌로만 좁혔기에, 100장이 다 나간 순간부터 나머지 요청은 즉시 실패로 떨어진다. "정확히 100명만 성공"은 충돌은 흡수하고 비즈니스 실패는 즉시 전파하는 이 구분에서 나온다.

### 비관 vs 낙관: 같은 테스트, 다른 비용

두 경로는 같은 동시성 테스트를 미러링한다. 그래서 결과 불변식(성공 100 / 잔여 0 / 발급행 100)은 똑같이 성립한다. 다른 건 그 100건을 만드는 비용이다.

이를 눈으로 보려고 낙관 경로 테스트는 성공한 발급의 시도 횟수를 함께 모은다. 발급 결과에 그 발급이 몇 번째 시도에서 성공했는지(`attempts`, 첫 시도 성공이면 1)를 실어, 거기서 1을 뺀 재시도 횟수를 100건에 걸쳐 합산한다. 충돌 없이 한 번에 성공하면 0, 충돌로 한 번 더 돌면 1씩 늘어난다. 동시 발사 직전부터 전부 완료까지의 wall-clock도 함께 잰다.

동일 조건(재고 100, 서로 다른 2,000명 동시 발급)에서 관측한 값은 이랬다.

- **낙관 락**: 성공 100건당 누적 재시도 약 58~121회, 소요 약 430~690ms. 모든 불변식 충족.
- **비관 락**: 충돌·재시도 0. 같은 재고 행을 두고 락 대기로 직렬화.

여기서 눈여겨볼 건 낙관 락 수치가 실행마다 출렁인다는 점이다. 같은 테스트를 반복해 돌리면 누적 재시도가 58회였다가 75회, 121회로 매번 달라진다. 2,000개의 가상 스레드가 같은 재고 행에 몰리는 타이밍이 실행마다 다르고, 그 미세한 차이가 충돌 횟수를 흔들기 때문이다. 비관 락이라면 이 값이 언제나 0으로 고정인 것과 정반대다. 이 변동성 자체가 낙관 락의 성격을 보여준다 — 비용이 경합의 운에 좌우된다.

해석하면, 낙관 락은 100장을 발급하면서 그 위에 수십~백여 번의 헛된 시도를 더 치른다. 경합이 심할수록 같은 버전을 읽고 동시에 커밋하려는 트랜잭션이 많아져 충돌이 잦아지고, 재시도 횟수가 늘어난다. 반면 비관 락은 애초에 한 번에 하나만 재고 행을 잡으므로 충돌이 없는 대신, 나머지는 락 앞에서 대기한다.

지난 글에서 "비관 락은 대기 비용을, 낙관 락은 재시도 비용을 치른다"고 적었는데, 이번에 그 재시도 비용을 숫자로 확인한 셈이다. 선착순 쿠폰처럼 하나의 행에 경합이 극심하게 쏠리는 상황에서는 낙관 락이 불리하다. 충돌 확률이 높아 재시도가 많아지고, 그 재시도는 전부 데이터베이스 왕복을 다시 하는 비용이기 때문이다. 낙관 락이 빛나는 곳은 반대로 충돌이 드문 경우 — 평소엔 락 비용을 전혀 안 내다가 가끔 부딪힐 때만 다시 시도하면 되는 워크로드다.

## 구현의 한계 / 문제점

### 경합이 극심한 선착순에는 낙관 락이 적합하지 않다

위 숫자가 그대로 보여준다. 100장을 두고 2,000명이 동시에 같은 행을 갱신하려 하니 충돌이 잦고, 성공 100건당 수십 번의 재시도가 붙는다. 재고 한 행에 쏠리는 경합 구조에서는 비관 락의 대기가 낙관 락의 재시도보다 대체로 싸다. 두 경로를 만들어 비교한 결론이 "이 문제에는 낙관 락을 쓰지 말라"는 쪽으로 기우는 것 자체가 이번 실험의 수확이다.

### `MAX_ATTEMPTS = 100`은 근거가 약한 상수다

관측된 재시도가 성공 100건 합산 58~121회였으니 개별 요청이 100번까지 가는 일은 이번 부하에선 없었지만(누적값이 100을 넘어도 그건 100건에 분산된 합이지 한 요청이 100번 진 게 아니다), 이 한도는 부하가 더 커지면 어떻게 움직이는지를 측정해 정한 값이 아니다. 경합이 더 심해지면 일부 요청이 한도를 다 쓰고도 성공하지 못해 충돌 예외로 떨어질 수 있다. 그건 재고가 없어서가 아니라 "운이 나빠 계속 졌기" 때문인데, 호출하는 쪽 입장에선 정당한 재고 소진 실패와 구분되지 않는다.

### 재시도에 백오프가 없다

루프는 충돌이 나면 곧바로 다음 시도로 넘어간다. 진 트랜잭션들이 쉼 없이 즉시 다시 부딪히므로, 경합이 심할 때 같은 충돌이 짧은 간격으로 반복되며 데이터베이스에 불필요한 부하를 더한다. 실제 운영이라면 시도 사이에 점증 지연(exponential backoff)이나 지터를 두어 동시 재시도가 한꺼번에 몰리지 않게 흩뜨려야 한다.

### 진짜 선착순(도착 순서)은 여전히 보장하지 않는다

이건 비관 경로와 공유하는 한계다. 낙관 락도 "초과 발급을 막는다"까지만 보장하지, "먼저 요청한 사람이 먼저 받는다"는 보장하지 않는다. 오히려 재시도가 끼면서 도착 순서와 성공 순서의 어긋남은 더 커질 수 있다. 충돌로 한 번 진 요청은 뒤로 밀려 다시 줄을 서기 때문이다.

## 다음에 해볼 내용

지금까지 동시성 제어를 전부 단일 데이터베이스 행 위에서 풀었다. 비관 락은 그 행을 잠그고, 낙관 락은 그 행의 버전으로 충돌을 감지한다. 둘 다 모든 요청이 결국 같은 한 행으로 수렴해 경합한다는 구조는 같다. 선착순의 본질적인 병목이 바로 여기다.

다음 글에서는 이 경합을 데이터베이스 바깥으로 끌어내는 접근을 같은 테스트로 견줘본다.

- 발급 가능 여부 판단을 인메모리 저장소의 원자적 연산으로 옮겨, 데이터베이스 행 경합을 먼저 걸러내는 방법.
- 그렇게 했을 때 새로 생기는 문제 — 인메모리 상태와 데이터베이스 사이의 정합성을 어떻게 맞출 것인가.
