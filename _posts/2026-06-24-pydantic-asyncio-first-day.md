---
title: "Pydantic과 AsyncIO, 손으로 만지며 배운 하루"
date: 2026-06-24 22:48:43 +0900
categories: [Backend, Python]
tags: [python, pydantic, asyncio, validation, async]
description: "Pydantic의 검증·coercion·직렬화와 AsyncIO의 gather·블로킹 함정·to_thread를, 실패하는 코드를 직접 돌려보며 정리한 학습 로그."
---

AI Robotics Engineer로 가는 첫 단계는 AI 백엔드 심화다. 그 시작이 Python 4종 세트(Pydantic, AsyncIO, SQLAlchemy, FastAPI)인데, 오늘 앞의 두 개를 끝냈다. 강의 정주행이 아니라 **코드를 직접 치고 결과(특히 에러와 실행 시간)를 눈으로 확인하는** 방식으로 갔다. 그 과정에서 의외로 조용한 버그도 하나 만났는데, 그게 가장 기억에 남는다.

---

## 1. Pydantic — "타입만 선언하면 검증이 끝난다"

### 왜 필요한가

백엔드를 하다 보면 외부에서 들어온 데이터(JSON, 폼, 설정 파일)를 dict로 받아서 `if not isinstance(x, int): raise ...` 같은 검증을 손으로 짜는 고통이 있다. Pydantic은 **클래스에 타입만 선언하면 검증·형변환·에러 메시지를 자동으로** 만들어 준다.

```python
from pydantic import BaseModel

class Robot(BaseModel):
    name: str
    battery: int

r1 = Robot(name="R2D2", battery=80)
print(r1)   # name='R2D2' battery=80
```

타입에 안 맞는 값을 넣으면?

```python
Robot(name="C3PO", battery="배터리없음")
```

```text
pydantic_core.ValidationError: 1 validation error for Robot
battery
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='배터리없음', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/int_parsing
```

직접 짰으면 만들었어야 할 친절한 에러 메시지(어느 필드가, 무슨 타입이어야 하는데, 무슨 값이 들어왔는지)를 공짜로 받는다.

### coercion — 숫자처럼 생긴 문자열은 통과한다

여기서 헷갈리기 쉬운 지점. `battery="80"`처럼 **숫자로 해석 가능한 문자열**을 넣으면 에러가 아니라 자동으로 `80`(int)으로 바뀐다. 이걸 **coercion(형 강제변환)**이라 한다.

- `"80"` → `80` ✅
- `"배터리없음"` → 변환 불가 → `ValidationError`

### Optional과 기본값

"필수 vs 선택" 필드는 이렇게 표현한다.

```python
class Robot(BaseModel):
    name: str                    # 필수 (기본값 없음)
    battery: int = 100           # 기본값 → 안 주면 100
    nickname: str | None = None  # 있어도 되고 없어도 됨

Robot(name="R2D2")  # battery=100, nickname=None 으로 채워짐
```

`nickname: str | None = None`에서 **`= None`을 지우면** `nickname`은 필수가 된다. `| None`은 "None이 들어올 수 있다"는 타입 얘기일 뿐, "안 줘도 된다"는 기본값 얘기와는 다르다.

### 오늘의 함정 — 조용히 사라진 값

실습 중에 필드 이름을 `battery`가 아니라 `beterry`로 오타를 냈는데, 정의와 생성자 양쪽에 똑같이 오타가 나서 코드는 **그냥 잘 돌았다.** 문제는 그다음이었다.

```python
class Robot(BaseModel):
    name: str
    beterry: int = 100   # 오타로 정의됨

r = Robot(name="Robo2", battery=50)   # 나는 50을 넣었다고 생각했다
print(r)   # name='Robo2' beterry=100  ← 50이 사라졌다!
```

왜 이런 일이? Pydantic v2는 기본적으로 **모르는 키(`battery`)를 조용히 무시**한다. 그래서 진짜 필드 `beterry`는 값이 없는 걸로 처리돼 **기본값 100**으로 채워졌다. 나는 50을 넣었다고 믿지만 실제로는 100이 들어간 것이다. 에러도, 경고도 없다. 운영에서 이런 게 로그도 안 남는 조용한 버그가 된다.

막는 방법은 한 줄이다.

```python
from pydantic import BaseModel, ConfigDict

class Robot(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 키가 오면 에러
    name: str
    battery: int = 100
```

이러면 `Robot(name="X", batery=50)` 같은 오타가 **입력 단계에서 즉시 `ValidationError`**로 터진다. 교훈: 이름은 정확히 일치해야 채워지고, 실무에서는 `extra="forbid"`로 오타를 일찍 잡는 게 안전하다.

### 중첩 모델과 파싱

실무 데이터는 JSON처럼 모델 안에 모델이 들어간다. 외부 입력은 보통 dict로 오니 `model_validate`로 통째로 파싱한다.

```python
class Battery(BaseModel):
    level: int
    charging: bool = False

class Robot(BaseModel):
    name: str
    battery: Battery   # 필드 타입이 또 다른 모델

data = {"name": "R2D2", "battery": {"level": 80, "charging": True}}
r = Robot.model_validate(data)
print(r.battery.level)   # 80
```

여기서 좋았던 점: 중첩이 깊어져도 에러는 **정확한 위치를 점 표기로** 짚어준다. 예를 들어 `battery`의 `level`에 잘못된 값을 주면 에러가 `battery` 전체가 아니라 `battery.level`을 가리킨다. 수동 검증으로는 "어디서 틀렸는지"를 만드는 게 제일 귀찮은데, 이걸 공짜로 받는다.

### 직렬화 — 들어올 땐 validate, 나갈 땐 dump

반대로 모델을 다시 밖으로 내보내는 게 직렬화다.

```python
r.model_dump()        # {'name': 'R2D2', 'battery': {'level': 80, 'charging': True}}  ← dict
r.model_dump_json()   # '{"name":"R2D2","battery":{"level":80,"charging":true}}'      ← str
```

- 중첩 모델은 **재귀적으로** 중첩 dict/JSON으로 풀린다.
- `model_dump()`는 Python 객체(dict, `True`), `model_dump_json()`은 JSON 문자열(`true` 소문자, 따옴표).

**들어올 땐 `model_validate`, 나갈 땐 `model_dump`.** 이 왕복이 Pydantic의 전부다. 그리고 이게 중요한 이유는, 나중에 FastAPI 응답·DB 저장·로깅이 전부 이 직렬화로 나가기 때문이다.

---

## 2. AsyncIO — "기다리는 시간에 다른 일을 한다"

### 왜 필요한가

네트워크·DB 같은 I/O를 기다리는 동안 CPU는 논다. async는 그 **대기 시간에 다른 일**을 시킨다. 나중에 FastAPI 요청 처리나 LLM 호출이 전부 이 위에서 돈다.

### 순차 vs 동시 — 시간으로 체감하기

`asyncio.sleep`으로 가짜 I/O를 만들어 실행 시간을 직접 쟀다.

```python
import asyncio, time

async def fake_io(name, seconds):
    print(f"{name} 시작")
    await asyncio.sleep(seconds)
    print(f"{name} 완료")

async def main():
    # 순차
    start = time.perf_counter()
    await fake_io("A", 1)
    await fake_io("B", 2)
    await fake_io("C", 3)
    print(f"순차: {time.perf_counter() - start:.2f}초")   # 6.00초

    # 동시
    start = time.perf_counter()
    await asyncio.gather(fake_io("A", 1), fake_io("B", 2), fake_io("C", 3))
    print(f"동시: {time.perf_counter() - start:.2f}초")   # 3.00초

asyncio.run(main())
```

결과:

- **순차 = 1+2+3 = 6초**
- **동시(gather) = max(1,2,3) = 3초**

핵심 직관: `gather`는 셋을 동시에 "기다리기 시작"하므로 전체 시간은 합이 아니라 **가장 오래 걸리는 것(max)**이다. 그리고 동시 실행에서는 "A,B,C 시작"이 먼저 몰려 찍히고 완료가 나중에 찍힌다. `await`를 만나면 이벤트 루프가 **제어권을 다른 코루틴에 넘기기** 때문이다.

### 오늘의 함정 — gather인데 안 빨라진다

딱 한 군데만 바꿔봤다. `await asyncio.sleep` 대신 **`time.sleep`**(블로킹).

```python
async def bad_io(name, seconds):
    print(f"{name} 시작")
    time.sleep(seconds)        # ← await가 없는 블로킹 호출
    print(f"{name} 완료")

# gather로 묶었는데도...
```

결과는 충격적이었다. print가 "A 시작 → A 완료 → B 시작 → ..."처럼 **완전히 순차로** 찍히고, 시간도 다시 **6초**로 돌아갔다. 분명 `gather`로 동시 실행을 시켰는데도.

이유: `time.sleep`은 `await`가 아니라서 이벤트 루프에게 제어권을 안 넘긴다. 자는 동안 **루프 전체를 멈춰 세운다.** gather로 묶어도 한 놈이 자는 동안 아무도 못 움직인다. 이게 실무에서 "async로 짰는데 왜 안 빨라지지?"의 90% 원인이다 — `requests`(동기 HTTP), 동기 DB 드라이버, `time.sleep`, 무거운 동기 라이브러리를 코루틴 안에서 그냥 호출하면 async가 무력화된다.

### 구출 — `asyncio.to_thread`

해결도 한 줄이었다.

```python
await asyncio.to_thread(time.sleep, seconds)   # 블로킹 함수를 별도 스레드로 떠넘김
```

다시 **3초**로 돌아왔고, print도 "시작 몰림" 패턴으로 복귀했다. `to_thread`는 블로킹 함수를 다른 OS 스레드에서 돌리고 `await`로 결과를 기다리므로, 이벤트 루프 본인은 안 멈춘다.

### `await (코루틴)` vs `to_thread` — 둘 다 비동기 같은데?

오늘 가장 헷갈렸고, 그래서 가장 명확해진 부분.

주방에 요리사 한 명(= 메인 스레드)이 있다고 하자.

- **`await (코루틴)`** = 그 한 명이 냄비 A를 올려두고 끓는 동안 B·C를 번갈아 본다. **새 사람을 안 뽑는다.** 단일 스레드 안에서 협력적으로(cooperative) 양보할 뿐이다. 이게 되려면 코드가 `await` 지점에서 **스스로 양보**할 줄 알아야 한다.
- **`to_thread`** = 양보할 줄 모르는 일을 **보조 요리사(별도 스레드)**에게 통째로 떠넘긴다. 동기 코드는 스스로 양보를 못 하니, 강제로 떼어내는 것이다.

선택 기준:

| 상황 | 방법 | 이유 |
|---|---|---|
| 라이브러리가 async 지원(awaitable): `httpx`, `asyncpg` | `await` | 정석. 스레드 오버헤드 없어 더 효율적 |
| 동기밖에 없음: `requests`, 동기 DB 드라이버, `time.sleep`, 파일 IO | `to_thread` | 양보 못 하는 코드를 스레드로 떠넘김 |

한 문장으로: **awaitable이 있으면 `await`가 정석, 동기뿐이면 `to_thread`로 구출.**

> 보너스: `to_thread`는 I/O 블로킹엔 잘 듣지만, 순수 CPU 무거운 계산엔 GIL 때문에 한계가 있다(그건 `ProcessPoolExecutor` 영역). CPU 작업은 또 다른 얘기다.

---

## 정리

**Pydantic** — 모델 정의 → 검증/coercion → Optional·기본값 → 중첩 → `model_validate`(파싱) → `model_dump`(직렬화). 들어올 땐 validate, 나갈 땐 dump.

**AsyncIO** — `async`/`await`(단일 스레드 협력) → `gather`(동시 대기, 시간=max) → blocking 함정(양보 안 하면 루프가 멈춤) → `to_thread`(동기 코드 구출).

두 주제를 관통하는 건 "이론을 외우는 대신 실패하는 코드를 직접 돌려보는 것"이었다. `beterry`로 50이 사라지는 걸 직접 본 것, gather인데 6초가 나오는 걸 직접 잰 것 — 이 두 장면이 어떤 설명보다 오래 남을 것 같다.

다음은 **SQLAlchemy**. 오늘 다진 "동기 vs async" 감각이 동기 엔진 vs `create_async_engine` 선택으로 그대로 이어진다.
