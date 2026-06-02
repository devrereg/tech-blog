---
title: "쿠폰발급 시스템으로 동시성 제어 구현하기(2) — 락 없는 쿠폰발급의 lost update를 테스트로 재현하기"
date: 2026-06-03 01:44:51 +0900
categories: [Backend, Concurrency]
tags: [concurrency, coupon, virtual-threads, lost-update, kotlin]
description: "락 없는 선착순 쿠폰 발급이 동시 요청에서 한도를 초과 발급함을, 가상 스레드 2,000개를 동시에 푸는 테스트로 재현한다."
---

> 락이 없이 쿠폰발급 서비스가 동시 요청에서 실제로 한도를 초과 발급함을 가상 스레드 2,000개를 동시에 실행하는 테스트로 재현했다. (2026-06-03)

## 구현 내용

[지난 글]({% post_url 2026-06-02-coupon-concurrency-1-baseline %})에서 락 없이 만든 선착순 쿠폰 발급 서비스가 단일 스레드에서는 정확하지만 동시 요청에서 lost update(갱신 분실)로 깨진다고 예측했다. 이번에는 그 예측을 다중 스레드 테스트로 확인했다.

테스트의 시나리오는 단순하다. **재고 100개짜리 쿠폰 하나에 서로 다른 사용자 2,000명이 같은 순간에 발급을 요청**한다. 그리고 발급이 끝난 뒤 "올바른 상태"를 단언한다 — 성공은 정확히 100건, 잔여 재고는 0, 발급 이력 행도 100개여야 한다.

```kotlin
// 올바른 불변식
assertThat(successCount).isEqualTo(TOTAL_QUANTITY)   // 100
assertThat(remaining).isEqualTo(0)
assertThat(issued).isEqualTo(TOTAL_QUANTITY.toLong()) // 100
```

동시 발사의 핵심은 모든 스레드를 출발선에 모았다가 한 번에 푸는 것이다. `CountDownLatch` 배리어 세 개(ready/start/done)로 이를 구현했다.

```kotlin
val ready = CountDownLatch(CONCURRENCY)
val start = CountDownLatch(1)
val done = CountDownLatch(CONCURRENCY)

Executors.newVirtualThreadPerTaskExecutor().use { executor ->
    userIds.forEach { userId ->
        executor.submit {
            ready.countDown()          // 준비 완료 알림
            start.await()              // 발사 신호 대기
            try {
                service.issue(couponId, userId)
                success.incrementAndGet()
            } catch (e: Exception) {
                fail.incrementAndGet()
            } finally {
                done.countDown()
            }
        }
    }

    ready.await(READY_TIMEOUT_SECONDS, TimeUnit.SECONDS) // 모두 출발선에 설 때까지
    start.countDown()                                    // 동시 발사
    check(done.await(DONE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
        "동시 발급이 ${DONE_TIMEOUT_SECONDS}s 안에 끝나지 않았다"
    }
}
```

## 구현 내용 설명

이 테스트를 쓰면서 내린 결정 네 가지가 학습 핵심이다. 하나씩 왜 그렇게 했는지 정리한다.

### 1. 왜 `@SpringBootTest`인가 — 트랜잭션으로 감싸면 안 된다

Spring에서 JPA 영속 계층을 테스트할 때 손이 먼저 가는 건 `@DataJpaTest`다. 슬라이스만 띄워 빠르고, 테스트가 끝나면 자동으로 롤백해줘서 데이터 정리도 필요 없다. 그런데 이 "자동 롤백"이 정확히 이번 재현을 불가능하게 만든다.

`@DataJpaTest`는 **테스트 메서드 전체를 하나의 트랜잭션으로 감싼다.** 그러면 테스트 안에서 일어나는 모든 작업이 같은 트랜잭션·같은 영속성 컨텍스트를 공유하고, 끝에서 통째로 롤백된다. lost update는 **여러 트랜잭션이 각자 독립적으로 커밋하면서** 서로의 갱신을 덮어쓸 때 생기는 현상인데, 모두가 한 트랜잭션 안에 있으면 애초에 경쟁할 트랜잭션 경계가 없다.

그래서 `@SpringBootTest`를 썼다. 이건 테스트 메서드를 트랜잭션으로 감싸지 않는다. 각 스레드가 호출하는 발급 메서드의 `@Transactional`이 **저마다 독립적으로 시작되고 독립적으로 커밋**된다. 이게 운영 환경에서 실제로 일어나는 일과 같다.

대신 데이터 정리가 필요하다. 자동 롤백이 없으니 커밋된 데이터가 그대로 남는다. 테스트마다 직접 비워야 하는데, 외래 키 때문에 삭제 순서가 정해진다 — 자식 행부터 지운다.

```kotlin
@AfterEach
fun tearDown() {
    // FK 순서: couponIssue → stock → coupon, user (자식부터)
    couponIssueJpaRepository.deleteAllInBatch()
    couponStockJpaRepository.deleteAllInBatch()
    couponJpaRepository.deleteAllInBatch()
    userJpaRepository.deleteAllInBatch()
}
```

발급 이력이 쿠폰·사용자를 참조하고, 재고가 쿠폰을 참조한다. 그래서 발급 이력 → 재고 → 쿠폰 → 사용자 순서로 지워야 외래 키 위반 없이 정리된다.

### 2. 왜 그냥 submit이 아니라 배리어인가

스레드 2,000개를 만들어 곧장 작업을 제출하면, 먼저 제출된 작업이 먼저 시작해 먼저 끝나는 경향이 생긴다. 작업이 줄지어 흘러가면 동시에 같은 재고 행을 읽는 일이 잘 일어나지 않아 경쟁이 약해진다. lost update를 보려면 **여러 스레드가 차감 직전 같은 잔여 수량을 동시에 읽는 순간**을 최대한 많이 만들어야 한다.

그래서 출발선을 둔다. 각 스레드는 일을 시작하기 전에 `ready.countDown()`으로 "준비됐다"를 알리고 `start.await()`에서 멈춰 선다. 메인 스레드는 2,000개가 전부 출발선에 설 때까지(`ready.await()`) 기다렸다가 `start.countDown()` 한 번으로 동시에 푼다. 마지막 `done` 래치는 모든 스레드가 끝났는지 확인하고 결과를 읽는 시점을 맞춘다.

스레드 2,000개를 이렇게 가볍게 다룰 수 있는 건 Java 21의 가상 스레드(Virtual Threads) 덕이다. `newVirtualThreadPerTaskExecutor()`는 작업마다 가상 스레드를 하나씩 붙이는데, 가상 스레드는 OS 스레드가 아니라 JVM이 관리하는 경량 스레드라 수천 개를 띄워도 부담이 적다. 발급은 대부분 DB 입출력을 기다리는 작업이라, 블로킹 동안 OS 스레드를 점유하지 않는 가상 스레드와 특히 잘 맞는다.

### 3. 왜 사용자를 2,000명 다 다르게 했나

이 쿠폰은 1인 1매다 — 발급 이력에 `(쿠폰, 사용자)` 유니크 제약이 걸려 있다. 만약 같은 사용자 한 명이 2,000번을 동시에 요청하게 했다면, 두 번째 요청부터는 전부 이 유니크 제약에 막혀 실패했을 것이다. 그러면 관문이 두 개가 되어버린다 — 재고 경쟁과 중복 경쟁. 둘이 섞이면 "재고 차감이 덜 됐다"는 현상이 중복 실패에 가려 깨끗하게 안 보인다.

그래서 **사용자를 모두 다르게** 두어 1인 1매 제약을 비활성화했다. 2,000명 모두 자격이 있으니 막는 관문은 오직 재고 하나가 된다. 이렇게 해야 재고 차감의 lost update만 순수하게 관찰된다.

### 4. 의도적으로 깨지는 테스트를 기본 빌드에서 분리

이 테스트는 "올바른 상태"(성공 100, 잔여 0, 발급행 100)를 단언한다. 그런데 지금 발급 서비스에는 락이 없으니, 이 단언은 통과하지 못한다 — 그게 정상이다. 문제는 보통의 테스트였다면 `./gradlew build`의 테스트 단계에서 이게 실패해 빌드 전체가 깨진다는 것이다. 정상 코드의 회귀를 잡아야 할 빌드가, 아직 구현하지 않은 기능 때문에 항상 빨간불이 되면 곤란하다.

그래서 이 테스트에 `concurrency` 태그를 붙이고, 기본 테스트에서 제외했다.

```kotlin
tasks.test {
    useJUnitPlatform {
        excludeTags("concurrency")
    }
}

// 동시성 재현 테스트만 온디맨드로 실행: ./gradlew concurrencyTest
tasks.register<Test>("concurrencyTest") {
    useJUnitPlatform {
        includeTags("concurrency")
    }
    shouldRunAfter(tasks.test)
}
```

`./gradlew build`는 이 테스트를 건너뛰어 평소대로 통과하고, 동시성 깨짐을 보고 싶을 때만 `./gradlew concurrencyTest`로 따로 돌린다. 쿠폰발급 로직에 락을 넣어 이 테스트가 통과하게 되면, 태그를 떼고 기본 빌드에 편입해 회귀 테스트로 그대로 쓸 수 있다.

## 구현의 한계 / 문제점

테스트를 돌리면 예측대로 깨진다. 그리고 깨지는 규모가 생각보다 크다.

```text
./gradlew build           → BUILD SUCCESSFUL  (동시성 테스트 제외, green 유지)
./gradlew concurrencyTest → BUILD FAILED
    expected: 100 but was: 888
    동시성 결과 — 성공=888 실패=1112 잔여=0 발급행=888
```

재고 100개에 2,000명이 몰리자 **888건이 발급됐다.** 한도의 거의 9배다. 발급 서비스의 흐름을 다시 보면 차감이 이렇게 새는 지점이 정확히 보인다.

```kotlin
val stock = couponStockRepository.findByCouponId(couponId)  // (A) 잔여 수량 읽기
    ?: throw CouponNotFoundException(couponId)
stock.decrease()                                            // (B) 메모리에서 차감
```

차감 규칙을 가진 `decrease()` 자체는 멀쩡하다.

```kotlin
fun decrease() {
    if (remainingQuantity <= 0) {
        throw OutOfStockException()
    }
    remainingQuantity -= 1
}
```

문제는 서비스가 (A) 읽기와 (B) 차감을 락 없이 이어 붙인 **read-modify-write**라는 데 있다. 수백 개의 트랜잭션이 (A)에서 똑같이 `remainingQuantity = 100`을 읽는다. 100은 0보다 크니 `decrease()`의 가드를 전부 통과하고, 각자 99로 줄인 값을 커밋한다. 잔여가 0까지 내려가면 그 다음부터는 0을 읽은 트랜잭션들이 `decrease()`의 `<= 0` 가드에 막혀 실패하므로, 재고는 음수로 가지 않고 0에서 멈춘다 — 스키마에도 `CHECK (remaining_quantity >= 0)` 제약이 있지만, 애초에 음수 쓰기를 시도하는 트랜잭션이 없어 이 제약이 실제로 발동하지는 않는다. 그래서 한도 위반은 "음수 재고"가 아니라 **"발급 수가 한도를 넘는다"**로 새어 나온다. 정확히 지난 글에서 예측한 모양이다.

한 가지 짚을 점은, 깨졌어도 데이터 자체는 어긋나지 않았다는 것이다. **성공 카운트(888)와 발급 이력 행 수(888)가 정확히 일치**한다. 발급에 성공했다고 집계된 만큼 이력이 실제로 남았다는 뜻이다 — 잘못된 건 "몇 명에게 줬는가"이지 "집계와 실제가 따로 논다"는 게 아니다. 깨짐이 깨끗하게 재현됐다고 말할 수 있다.

`decrease()`가 단일 스레드에서 옳다고 동시성에서 옳은 게 아니다. 올바른 도메인 규칙도, 그걸 호출하는 쪽이 읽기와 쓰기 사이를 보호하지 않으면 무너진다.

### 이건 서버를 여러 대 돌려서 생긴 문제가 아니다

한 가지 오해를 짚고 가자. "동시성 문제는 서버를 여러 대 띄워야 생기는 것 아닌가?" — 아니다. 이 테스트가 그 증거다. 위 결과는 **서버 한 대도 아닌, 테스트 프로세스 하나** 안에서 스레드 2,000개를 푼 것만으로 나왔다.

동시성의 단위는 서버가 아니라 **스레드, 곧 트랜잭션**이다. 서버 한 대도 요청을 하나씩 줄 세워 처리하지 않는다. 스레드 풀(혹은 이 프로젝트처럼 요청마다 가상 스레드)로 수백·수천 개 요청을 동시에 처리한다. 요청 하나가 스레드 하나·트랜잭션 하나를 잡으니, 서버가 한 대여도 같은 재고 행을 동시에 읽는 트랜잭션은 얼마든지 생긴다. 경쟁이 일어나는 곳은 "서버"가 아니라 여러 트랜잭션이 함께 건드리는 **공유 상태(여기서는 재고 행)**다.

서버 대수가 바꾸는 건 문제의 유무가 아니라 **해결 방법의 선택지**다. 서버 한 대라면 JVM 안의 락(`synchronized` 등)으로 스레드를 직렬화하는 선택지가 있긴 하다. 하지만 서버를 두 대로 늘리는 순간, 각 서버는 별도 JVM이라 서로의 JVM 락을 알지 못해 그대로 깨진다. 그래서 공유 상태가 있는 곳 — DB 행 자체나, 여러 서버가 공유하는 분산 락 — 을 잠그는 쪽이 서버 대수와 무관하게 안전하다.

## 다음에 해볼 내용

이제 깨지는 모양과 규모를 숫자로 확인했으니, 다음 글에서는 이 read-modify-write 구간을 보호한다.

- 재고 행에 **비관적 락(pessimistic lock)**을 걸어 읽는 순간부터 차감·커밋까지 다른 트랜잭션을 막는 방법.
- **낙관적 락(optimistic lock)**으로 버전 충돌을 감지해 재시도하는 방법.
- 둘의 트레이드오프(락 경합 비용 vs 재시도 비용)를 같은 테스트로 비교한다.

락을 넣어 이 테스트가 통과하면, 태그를 떼고 기본 빌드의 회귀 테스트로 편입한다.
