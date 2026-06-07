---
title: "쿠폰발급 시스템으로 동시성 제어 구현하기(5) — 재고 판단을 Redis로 옮기기"
date: 2026-06-07 19:10:26 +0900
categories: [Backend, Concurrency]
tags: [concurrency, coupon, redis, lua-script, kotlin]
description: "재고 판단을 DB 밖 Redis 카운터 + Lua 스크립트로 옮겨, DB 재고 한 줄을 두고 다투던 경합 자체를 없앤다. 같은 동시성 테스트(재고 100, 2,000명)에서 정확히 100건만 발급하면서 153ms로 가장 빠르지만, 대신 Redis와 DB가 어긋날 수 있다는 새 문제를 떠안는다."
---

> 100장 한정 쿠폰에 2,000명이 동시에 몰리면 어떻게 정확히 100장만 나눠줄까. 앞선 글들은 이 문제를 데이터베이스 안에서 풀었다. 이번에는 "재고가 남았는지" 판단하는 일을 데이터베이스 밖 Redis로 빼냈다. 훨씬 빨라졌지만, 대신 새로운 골칫거리가 하나 생긴다. (2026-06-06)

## 무슨 문제를 풀고 있었나

선착순 쿠폰의 핵심 규칙은 단순하다. 100장이면 딱 100명까지만. 101명째는 "재고 소진"으로 막아야 한다.

어려운 건 "동시에" 몰릴 때다. 2,000명이 같은 순간에 발급 버튼을 누르면, 서버는 2,000개의 요청을 한꺼번에 처리하게 된다. 이때 자칫하면 100장을 두고 여러 요청이 "아직 남았네?" 하고 동시에 집어가서 101장, 120장이 나가버린다. 이걸 막는 게 동시성 제어다.

[지난 글]({% post_url 2026-06-05-coupon-concurrency-4-optimistic-lock %})까지 두 가지 방법을 만들어봤다.

- **방법 1 ([비관 락]({% post_url 2026-06-04-coupon-concurrency-3-pessimistic-lock %}))**: 재고를 건드리려는 요청들을 한 줄로 세운다. 한 명이 재고를 확인하고 깎는 동안 나머지는 문 앞에서 기다린다. 안전하지만 줄 서느라 느리다.
- **방법 2 ([낙관 락]({% post_url 2026-06-05-coupon-concurrency-4-optimistic-lock %}))**: 일단 다 같이 들어가서 깎되, "내가 본 사이에 누가 먼저 깎았나?"를 커밋 순간에 확인한다. 먼저 깎인 걸 발견한 요청은 다시 처음부터 시도한다. 줄은 안 서지만 충돌이 잦으면 재시도가 폭증한다.

두 방법은 겉보기엔 달라도 약점이 똑같다. 둘 다 결국 **데이터베이스의 재고 한 줄(row)** 을 모두가 동시에 만지려고 다툰다. 2,000명이 같은 한 줄에 몰리니, 줄을 서든(방법 1) 충돌하고 재시도하든(방법 2) 비용을 치른다.

이번 글의 아이디어는 그 다툼의 장소 자체를 옮기는 것이다.

## 핵심 아이디어 — 재고 판단을 "번호표 기계"로

비유하면 이렇다. 지금까지는 직원 한 명이 장부(데이터베이스)에 적힌 재고 숫자를 보고 손님마다 직접 깎아줬다. 손님이 2,000명이면 그 장부 한 줄 앞에서 병목이 생긴다.

대신 입구에 번호표 뽑는 기계를 둔다고 해보자. 기계 안에 번호표 100장을 채워두고, 손님은 들어오면서 한 장씩 뽑는다. 100장이 다 떨어지면 기계는 더 안 준다. 이 기계는 한 번에 한 사람씩만 번호표를 내주도록 만들어져 있어서, 두 사람이 같은 번호표를 받는 일이 없다.

이 "번호표 기계" 역할을 Redis의 숫자 카운터가 한다. Redis는 메모리에서 도는 빠른 저장소다. 재고를 이 카운터에 넣어두고, 발급 요청이 오면:

1. 먼저 Redis 카운터를 1 깎아본다 → 깎였으면 "통과"(번호표 받음), 0이라 못 깎으면 "재고 소진"으로 즉시 거절.
2. 통과한 요청만 데이터베이스에 "누가 발급받았다"는 기록을 남긴다.

여기서 중요한 건, 이제 데이터베이스 재고 한 줄을 두고 다투지 않는다는 점이다. 통과 판정은 Redis가 빠르게 내려주고, 통과한 요청들이 데이터베이스에 쓰는 건 **각자 자기 발급 기록 한 줄** 이라 서로 안 부딪힌다. 100명이 100개의 서로 다른 줄을 만들 뿐이다.

## 두 명이 같은 번호표를 받으면? — 한 번에 한 명만 통과시키기

여기가 이 글에서 가장 중요한 부분이다.

순진하게 만들면 이렇게 된다 — "재고를 본다 → 0보다 크면 깎는다"를 Redis에 두 번 나눠서 물어보는 것이다. 그런데 그 사이에 다른 요청이 끼어들 수 있다.

```text
[손님 A] 재고 봤더니 1 남음
[손님 B] 재고 봤더니 1 남음   ← A가 아직 안 깎았을 때 끼어듦
[손님 A] 깎음 → 0
[손님 B] 깎음 → -1            ← 둘 다 통과! 초과 발급
```

데이터베이스 한 줄에서 봤던 그 문제가 Redis로 자리만 옮겨 똑같이 재현된다. "보는 것"과 "깎는 것"이 쪼개지는 순간 틈이 생긴다.

해법은 둘을 절대 쪼개지지 않게 한 덩어리로 묶는 것이다. Redis에는 이걸 위한 도구가 있다 — **Lua 스크립트**다.

### 잠깐, Lua 스크립트가 뭔데?

Lua는 가벼운 프로그래밍 언어인데, Redis 안에 끼워 넣어 쓸 수 있다. 우리가 Lua로 짧은 코드를 적어 Redis에 보내면, Redis는 그 코드를 한 번에, 중간에 다른 요청을 끼워넣지 않고, 통째로 실행한다. (Redis는 명령을 한 줄로 처리하는 단일 스레드 구조라 이게 가능하다.)

즉 "재고 보고 → 깎기"를 Lua 한 덩어리로 보내면, 손님 A의 (보기→깎기)가 끝나기 전엔 손님 B가 절대 못 끼어든다. 이게 **원자적(atomic)** 이라는 말의 뜻이다. 쪼갤 수 없는 하나의 동작.

우리가 쓴 스크립트는 이렇게 짧다.

```lua
-- KEYS[1] = coupon:stock:{couponId}  (재고를 담아둔 카운터)
local remaining = tonumber(redis.call('GET', KEYS[1]))  -- ① 재고를 읽는다
if remaining == nil or remaining <= 0 then              -- ② 없거나 0 이하면
  return -1                                             --    안 깎고 -1 반환 (소진)
end
return redis.call('DECR', KEYS[1])                      -- ③ 남았으면 1 깎고 남은 값 반환
```

읽고(①), 남았는지 보고(②), 남았을 때만 깎는다(③). 이 셋이 통째로 한 번에 돈다. 재고가 0이면 ③에 도달하지 못하므로 카운터가 음수로 내려가는 일이 아예 없다.

실제로 Redis에 재고 2를 넣고 이 스크립트를 반복 실행해보면 반환값이 `1 → 0 → -1`로 가고, 그 뒤로는 계속 `-1`을 주며 카운터는 0에서 멈춘다. 의도한 그대로다.

## 코드로 보는 전체 흐름

번호표 기계(Redis 게이트)를 코드에서는 "재고 1단위를 예약한다"는 인터페이스로 추상화했다. 발급 로직은 이게 Redis로 만들어졌다는 걸 몰라도 된다.

```kotlin
interface StockReservation {
    fun prepare(couponId: Long, quantity: Int)   // 카운터를 재고 수량으로 초기화
    fun reserve(couponId: Long): Boolean         // 1장 예약 시도 (성공 true / 소진 false)
    fun release(couponId: Long)                  // 예약 취소 — 깎은 걸 되돌린다
    fun remaining(couponId: Long): Long          // 현재 잔여
}
```

실제 Redis 구현은 위 Lua를 실행하고, 반환값 -1(소진)을 false로 바꿔준다.

```kotlin
@Component
class RedisStockReservation(
    private val redisTemplate: StringRedisTemplate,
) : StockReservation {

    private val reserveScript = DefaultRedisScript<Long>().apply {
        setLocation(ClassPathResource("redis/reserve_stock.lua"))
        resultType = Long::class.java
    }

    override fun reserve(couponId: Long): Boolean {
        val remaining = redisTemplate.execute(reserveScript, listOf(key(couponId)))
        return remaining != null && remaining >= 0   // 0 이상이면 통과(0 = 마지막 한 장을 방금 확보), -1이면 소진
    }

    override fun release(couponId: Long) {
        redisTemplate.opsForValue().increment(key(couponId))  // 다시 1 올려 되돌림
    }

    private fun key(couponId: Long) = "coupon:stock:$couponId"
}
```

발급 요청 하나는 세 단계로 처리된다 — ① 게이트 통과 → ② 데이터베이스에 기록 → ③ (②가 실패하면) 되돌리기. 이 조율을 맡는 서비스다.

```kotlin
@Service
class RedisCouponIssueService(
    private val reservation: StockReservation,
    private val persist: RedisCouponIssuePersist,
) {
    fun issue(couponId: Long, userId: Long): CouponIssue {
        if (!reservation.reserve(couponId)) {     // ① 번호표 못 받으면
            throw OutOfStockException()           //    여기서 끝 (재고 소진)
        }
        try {
            return persist.persist(couponId, userId)  // ② 발급 기록 저장
        } catch (e: Exception) {
            reservation.release(couponId)         // ③ 저장 실패 → 번호표 반납
            throw e
        }
    }
}
```

데이터베이스 저장은 별도 빈이 맡고, 이 빈은 재고를 건드리지 않는다. 사용자 확인과 중복 확인만 하고 발급 기록을 남긴다. (재고는 이미 Redis가 책임지니까.)

```kotlin
@Service
class RedisCouponIssuePersist(
    private val userRepository: UserRepository,
    private val couponIssueRepository: CouponIssueRepository,
) {
    @Transactional
    fun persist(couponId: Long, userId: Long): CouponIssue {
        userRepository.findById(userId) ?: throw UserNotFoundException(userId)
        if (couponIssueRepository.existsByCouponIdAndUserId(couponId, userId)) {
            throw DuplicateIssueException(couponId, userId)
        }
        return couponIssueRepository.save(CouponIssue(couponId = couponId, userId = userId))
    }
}
```

## 번호표는 받았는데 발급이 실패하면? — 깎은 재고를 되돌리기

번호표를 받았는데(Redis 카운터는 이미 1 깎였는데) 그다음 데이터베이스 저장이 실패하면 어떻게 될까? 예를 들어 이미 발급받은 사람이 또 요청한 경우다.

아무것도 안 하면 번호표 한 장이 영영 증발한다. 실제로 발급은 안 됐는데 재고만 1 줄어든 상태 — 100장 중 한 장이 유령이 된다. 그래서 저장이 실패하면 `release`로 카운터를 1 되돌려(반납) 슬롯을 살려둔다. 위 코드의 ③번이 그 일이다.

여기서 한 가지 설계 결정이 숨어 있다. 되돌리기를 **"데이터베이스 트랜잭션 바깥"에서 해야 한다**는 점이다.

쉽게 말하면, "저장이 성공했는지 실패했는지"는 트랜잭션이 완전히 끝나봐야(커밋 또는 롤백) 안다. 그런데 그 끝나는 시점은 `@Transactional`이 붙은 메서드를 빠져나가는 순간이다. 그래서 되돌리기 판단은 그 메서드 바깥, 즉 "저장을 호출한 쪽"에서 결과를 받아본 뒤에 해야 한다.

이 때문에 빈을 둘로 나눴다 — 트랜잭션이 없는 조율 서비스(`RedisCouponIssueService`)와, 트랜잭션이 있는 저장 빈(`RedisCouponIssuePersist`). 조율 서비스가 저장 빈을 호출하면 트랜잭션은 그 호출에서 열리고 닫히고, 조율 서비스는 그게 예외로 끝났는지 보고 비로소 되돌린다.

(참고로 한 클래스 안에서 자기 메서드를 직접 부르면 `@Transactional`이 무시되는 함정이 있다. Spring이 프록시라는 대리자를 통해 트랜잭션을 거는데, 내부 호출은 그 대리자를 안 거치기 때문이다. 빈을 나눠 서로 호출하면 이 함정을 피한다. [낙관 락 글]({% post_url 2026-06-05-coupon-concurrency-4-optimistic-lock %})에서 다뤘던 것과 같은 이유다.)

## 결과 — 같은 100장, 더 싼 비용

세 방법 모두 같은 시험을 통과한다: 재고 100, 서로 다른 사용자 2,000명이 동시에 발급. 결과는 셋 다 정확히 100건 성공, 잔여 0. 다른 건 그 100장을 만드는 데 드는 비용이다.

Redis 게이트 방법이 남긴 로그는 이랬다.

```text
Redis 게이트 동시성 결과 — 성공=100 실패=1900 Redis잔여=0 발급행=100 소요=153ms
```

세 방법을 나란히 놓으면:

| 방법 | 줄 서기(락 대기) | 재시도 | 대략 소요 |
|------|------------------|--------|-----------|
| 비관 락 | 있음 (재고 줄에 줄 섬) | 없음 | 측정 안 함 |
| 낙관 락 | 없음 | 많음 (100건당 누적 58~121회) | 약 430~690ms |
| Redis 게이트 | 없음 | 없음 | 약 153ms |

낙관 락은 몰릴수록 재시도가 늘어 실행마다 비용이 출렁였고, 비관 락은 충돌은 없지만 줄을 세웠다. Redis 게이트는 다툼의 장소를 카운터로 옮겨 두 비용을 모두 안 낸다. 떨어진 1,900건도 게이트에서 -1을 받고 즉시 거절돼서 재시도도 대기도 없다.

## 빨라진 대가로 새로 생긴 문제 3가지

빠른 데는 대가가 있다.

### 1. 중간에 서버가 죽으면 번호표가 영영 사라진다

되돌리기는 "같은 요청이 저장 실패를 잡아서 `release`를 부르는" 구조다. 그런데 번호표를 받은 직후, 저장을 하기도 전에 서버가 죽어버리면? 되돌릴 코드가 영영 실행되지 않는다. 재고 한 장이 발급도 반납도 안 된 채 사라진다. 이건 "동기 호출 + 그 자리에서 되돌리기"라는 구조가 가진 근본적 한계다. 실제 운영이라면, 발급 의도를 먼저 어딘가에 적어두거나(아웃박스), 주기적으로 Redis와 데이터베이스를 대조해 어긋남을 메우는 장치가 필요하다.

### 2. 데이터베이스의 재고 숫자가 거짓말을 하게 된다

이 방법에서 재고를 깎는 건 오직 Redis 카운터다. 데이터베이스 테이블 `coupon_stocks`의 재고 컬럼(`remaining_quantity`)은 아무도 안 건드린다. 그래서 100장이 다 나가고 나면 두 곳의 숫자가 이렇게 갈라진다.

```text
Redis 카운터        : 0    ← 진짜 남은 수량 (다 나감)
데이터베이스 재고 컬럼 : 100  ← 처음 그대로 (안 줄었음)
```

이 방법 안에서는 일부러 이렇게 둔 것이다. 재고 판단은 Redis에게 다 맡기고, 데이터베이스는 "누가 받아갔나"를 적는 장부 역할만 하기로 했으니까. 데이터베이스 재고 컬럼은 더 이상 의미가 없는, 처음 시드값이 그대로 남은 죽은 숫자다.

문제는 이 사정을 모르는 다른 코드다. 가령 관리자 페이지나 다른 API가 "남은 수량 = 데이터베이스 재고 컬럼"이라고 믿고 읽으면, 실제로는 다 떨어졌는데도 화면엔 100이라고 뜬다.

여기서 얻는 교훈이 하나 있다. 같은 데이터(여기선 "재고")를 두 곳에 두면, "어느 쪽이 진짜냐"를 시스템 전체가 한 곳으로 약속해 둬야 한다. 이 약속을 흔히 **단일 진실 공급원(single source of truth)** 이라고 부른다. 이 경로에서 진짜는 Redis다. 그러니 "남은 수량"이 궁금한 코드는 전부 데이터베이스가 아니라 Redis를 봐야 한다. 한쪽은 진짜, 한쪽은 장부 — 이 역할 분담을 모두가 알고 있어야 사고가 안 난다.

### 3. 아직 실제로 쓸 준비는 안 됐다

이 경로는 내부 로직과 테스트까지만 만들어져 있다. 발급 시작 전에 카운터를 재고 수량으로 채우는 일(`prepare`)을 테스트에서는 직접 해주지만, 실제 발급 시작·종료에 맞춰 자동으로 채우고 비우는 흐름이나, 이 경로를 외부에 노출하는 REST 엔드포인트는 아직 없다.

## 다음에 해볼 내용

이번 글은 재고 판단을 데이터베이스 밖으로 빼서 빠르게 만들었지만, 대신 "Redis와 데이터베이스가 어긋날 수 있다"는 새 문제를 떠안았다. 다음 글에서는 이 어긋남을 줄이는 방향을, 지금까지 세 방법을 모두 통과시킨 그 동시성 테스트(재고 100에 서로 다른 2,000명이 동시 발급)로 똑같이 재서 견줘본다 — 잣대가 같아야 공정한 비교가 되니까.

- 발급 의도를 먼저 기록해두고, 데이터베이스 저장은 큐와 워커로 나중에 처리하는 방법.
- 또는 Redis 카운터와 발급 기록을 주기적으로 대조해 어긋남을 메우는 재동기화.
- 그리고 이 경로를 실제로 쓰려면 필요한 재고 시딩 흐름과 REST 연결.
