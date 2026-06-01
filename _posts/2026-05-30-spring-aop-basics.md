---
title: "Spring AOP 왜, 언제, 어떻게 사용하는가?"
date: 2026-05-30 03:56:49 +0900
categories: [Backend, Spring]
tags: [spring, aop, kotlin, annotation]
description: "관성적으로 써오던 Spring AOP를 왜·언제·어떻게 쓰는지 분산 락 예제로 정리한다."
---

## 들어가며

그동안 별생각 없이 관성적으로 사용하던 AOP를 한 번 정리해 본다.

## AOP란?

AOP는 **Aspect Oriented Programming**, 한국어로는 **관점 지향 프로그래밍**이다.
공통 기능(부가 기능)을 핵심 비즈니스 로직과 **분리해서** 관리하기 위한 프로그래밍 기법이다.

## 왜 사용하는가?

비즈니스 로직 안에 로깅, 분산 락 획득/반환, 트랜잭션, 권한 체크 같은 반복 코드가 섞이면 핵심 로직과 부가 기능 로직이 섞인다.

```kotlin
fun reserveTicket() {
    // 락 획득
    lock()

    // 티켓 예매 비즈니스 로직
    reserve()

    // 락 해제
    unlock()
}
```

문제는 명확하다.

- API가 늘어날 때마다 같은 코드가 반복된다.
- 매번 `lock()` / `unlock()`을 짝으로 호출해야 한다.
- `unlock()`을 빼먹으면 락이 풀리지 않는 버그가 생긴다.
- 핵심 로직이 부가 로직에 묻혀 가독성이 떨어진다.

AOP는 이 부가 기능을 분리해서 위 문제를 해결한다.

## 언제 사용하는가?

비즈니스 로직 **외부**에서 반복적으로 수행되는 작업, 또는 비즈니스 로직 **전/후**에 자동으로 끼워 넣고 싶은 로직이 있을 때 사용한다.

실무에서 자주 쓰는 곳:

- 로깅 / 메서드 실행 시간 측정
- 분산 락 획득·반환
- 트랜잭션 처리
- 권한 체크

## 어떻게 사용하는가?

분산 락을 예시로 단계별로 보자.

### 1. 의존성 추가

```kotlin
dependencies {
    implementation("org.springframework.boot:spring-boot-starter-aop")
}
```

### 2. 어노테이션 정의

```kotlin
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class DistributedLock(
    val key: String
)
```

| 코드 | 의미 |
| --- | --- |
| `@Target(FUNCTION)` | 메서드(함수)에만 붙일 수 있음 |
| `@Retention(RUNTIME)` | 런타임에도 어노테이션 정보 유지 (리플렉션으로 읽으려면 필수) |
| `val key: String` | 락 키 값 |

### 3. Aspect 클래스와 전/후 로직 작성

부가 로직은 `@Aspect`가 붙은 클래스의 메서드로 작성한다.

```kotlin
import org.aspectj.lang.ProceedingJoinPoint
import org.aspectj.lang.annotation.Around
import org.aspectj.lang.annotation.Aspect
import org.springframework.stereotype.Component

@Aspect
@Component
class DistributedLockAspect {

    @Around("@annotation(distributedLock)")
    fun lock(
        joinPoint: ProceedingJoinPoint,
        distributedLock: DistributedLock
    ): Any? {
        val key = distributedLock.key

        println("락 획득: $key")

        return try {
            joinPoint.proceed()
        } finally {
            println("락 해제: $key")
        }
    }
}
```

| 어노테이션 | 역할 |
| --- | --- |
| `@Aspect` | AOP 클래스임을 선언 |
| `@Component` | Spring Bean으로 등록 |

- `@Around("@annotation(distributedLock)")` — `@DistributedLock`이 붙은 메서드 호출을 가로챈다. 표현식의 `distributedLock`은 아래 메서드 파라미터명과 같아야 어노테이션 인스턴스가 주입된다.
- `joinPoint.proceed()` — 원래 메서드를 실제로 실행한다. 이 호출의 앞·뒤가 곧 "메서드 실행 전/후" 지점이다.
- `try-finally`로 감싸야 비즈니스 로직에서 예외가 나도 락이 반드시 풀린다.

### 4. 사용하기

```kotlin
@DistributedLock(key = "ticket")
fun reserveTicket() {
    reserve()
}
```

이제 `reserveTicket()`을 호출하면 Aspect가 알아서 락을 잡고 풀어준다. 호출부에는 `lock()` / `unlock()`이 보이지 않는다.

## 추가 정보 — Advice 종류

`@Around` 외에도 실행 시점에 따라 골라 쓸 수 있는 어노테이션이 있다.

| 어노테이션 | 실행 시점 |
| --- | --- |
| `@Before` | 메서드 실행 **전** |
| `@After` | 메서드 실행 **후** (성공/실패 무관) |
| `@AfterReturning` | 정상 반환 시 |
| `@AfterThrowing` | 예외 발생 시 |
| `@Around` | 전·후 모두 (직접 `proceed()` 호출) |

`@Around`를 많이 사용하지만, 단순 로깅·측정 정도면 `@Before`/`@AfterReturning`만 사용해도 되는 경우가 있다.

## 정리

- AOP는 **부가 기능을 비즈니스 로직에서 떼어내는** 도구다.
- 반복되는 로직, 메서드 전/후에 끼어드는 로직이 보이면 AOP를 의심해 본다.
- 직접 만들 일이 많진 않지만, Spring의 `@Transactional`이 어떻게 동작하는지 이해하는 기반이 된다.

## 참고

- [Spring Framework Reference — Aspect Oriented Programming](https://docs.spring.io/spring-framework/reference/core/aop.html)
