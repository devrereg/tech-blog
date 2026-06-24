---
title: "SQLAlchemy 2.0 async, 함정 두 개로 배운 ORM"
date: 2026-06-24 22:58:53 +0900
categories: [Backend, Python]
tags: [python, sqlalchemy, orm, async, asyncio]
description: "SQLAlchemy 2.0 + async에서 만난 두 함정(MissingGreenlet lazy loading, expire_on_commit)을, echo 로그로 SQL을 직접 관찰하며 정리한 학습 로그."
---

Python 4종 세트(Pydantic, AsyncIO, SQLAlchemy, FastAPI) 중 세 번째, SQLAlchemy를 끝냈다. 백엔드 경력이 있어서 SQL·트랜잭션·ORM 개념 자체는 익숙하다. 그래서 기초는 빠르게 넘기고, **SQLAlchemy 2.0 + async 고유의 함정**에 집중했다. 결론부터 말하면, 오늘도 가장 많이 배운 건 두 개의 에러 메시지였다. 하나는 `MissingGreenlet`, 하나는 commit 후에 객체가 조용히 죽는 문제.

실습은 로봇 도메인으로 갔다. `Robot` 1마리가 여러 개의 `TaskLog`(걷기, 집기 같은 작업 기록)를 갖는 1:N 관계. async SQLite(`aiosqlite`)로 띄웠다.

## 왜 async ORM인가

지난 글에서 AsyncIO를 "기다리는 시간에 다른 일을 한다"로 정리했다. DB 쿼리야말로 그 "기다리는 I/O"의 대표선수다. FastAPI 엔드포인트가 async로 돌아갈 거라면, 그 안에서 호출하는 DB도 async여야 이벤트 루프를 안 막는다. 동기 드라이버를 코루틴 안에서 그냥 부르면 — 지난 글의 `time.sleep` 함정처럼 — async가 무력화된다. 그래서 처음부터 `create_async_engine`으로 갔다.

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine("sqlite+aiosqlite:///robots.db", echo=True)
```

`echo=True`가 오늘의 숨은 주인공이다. SQLAlchemy가 **실제로 날리는 SQL을 콘솔에 전부 찍어준다.** ORM은 편한 대신 "내가 무슨 쿼리를 만드는지" 감이 사라지기 쉬운데, echo를 켜두면 그게 다 보인다. 오늘 배운 것의 절반은 이 로그를 눈으로 읽은 것이다.

## 모델 정의 — 2.0 스타일 (`Mapped` / `mapped_column`)

검색하면 옛날 문법(`Column(Integer, ...)`)이 많이 나오는데, 2.0부터는 타입 힌트 기반 스타일이 정석이다. Pydantic에서 타입 선언하던 감각이 그대로 이어진다.

```python
from datetime import datetime
from sqlalchemy import ForeignKey, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Robot(Base):
    __tablename__ = "robots"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    model: Mapped[str] = mapped_column(String(50))
    logs: Mapped[list["TaskLog"]] = relationship(back_populates="robot")

class TaskLog(Base):
    __tablename__ = "task_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    robot_id: Mapped[int] = mapped_column(ForeignKey("robots.id"))
    task: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    robot: Mapped["Robot"] = relationship(back_populates="logs")
```

포인트:
- `Mapped[int]`, `Mapped[str | None]` 같은 타입 힌트가 NULL 허용 여부까지 결정한다. `Mapped[str]`은 NOT NULL, `Mapped[str | None]`이면 nullable.
- `relationship(back_populates=...)`로 양쪽을 묶으면 `robot.logs`와 `log.robot`이 자동으로 동기화된다.
- `server_default=func.now()`는 기본값을 **DB가** 채우게 한다(애플리케이션이 아니라). echo 로그를 보면 INSERT문에는 `created_at`이 아예 빠져 있고(앱이 값을 안 보냄), 대신 테이블 DDL 쪽에 `created_at DATETIME DEFAULT CURRENT_TIMESTAMP`가 박혀 있다. DB가 채운 값은 `RETURNING created_at`으로 되읽어온다.

## Session 생명주기 — 그리고 `add` 한 줄의 마법

```python
async_session = async_sessionmaker(engine, expire_on_commit=False)

async with async_session() as session:
    r1 = Robot(name="Atlas", model="humanoid-v2")
    r1.logs = [TaskLog(task="walk"), TaskLog(task="pick", status="done")]
    session.add(r1)            # robot 하나만 add 했는데...
    await session.commit()
```

`session.add(r1)` **한 줄**만 했는데 echo 로그엔 INSERT가 세 번 찍혔다.

```
INSERT INTO robots ...        ('Atlas', 'humanoid-v2')
INSERT INTO task_logs ...     (1, 'walk', 'pending')
INSERT INTO task_logs ...     (1, 'pick', 'done')
```
(실제 echo 로그는 SQL과 바인딩 파라미터를 별도 줄로 찍지만, 여기선 읽기 편하게 합쳐 표기했다.)

robot만 저장했는데 task_log 2개까지 들어갔다. 이유는 **relationship의 cascade**. `relationship()`의 기본 cascade는 `save-update, merge`인데, 이 중 `save-update`가 작동해서 부모(`r1`)를 세션에 add하면 거기 매달린 자식(`r1.logs`)도 자동으로 저장 큐에 따라온다. 게다가 robot의 `id=1`을 받아서 task_logs의 `robot_id`에 알아서 채워 넣었다(외래키 자동 연결). 손으로 `robot_id=1`을 넣은 적이 없는데도.

`async_sessionmaker(...)`에 붙인 `expire_on_commit=False`는 잠시 뒤 함정 ②에서 설명한다. 일단 "async에선 거의 항상 이걸 끈다"만 기억하고 넘어갔다.

## 조회와 eager loading — N+1을 SQL 로그로 목격

조회는 `select()` + `session.execute()` 조합이다. 여기서 첫 번째로 발이 걸렸다.

```python
from sqlalchemy import select
from sqlalchemy.orm import selectinload

stmt = select(Robot).options(selectinload(Robot.logs))
result = await session.execute(stmt)
robots = result.scalars().all()   # ← 이게 핵심
```

처음에 `result.name`을 바로 찍으려다 `'ChunkedIteratorResult' object has no attribute 'name'`을 만났다. `session.execute()`가 돌려주는 건 **Robot이 아니라 결과를 꺼내는 커서**다. 거기서 실제 ORM 객체를 뽑는 게 `.scalars().all()`. 정리해두면:

| 메서드 | 의미 |
|---|---|
| `result.scalars().all()` | 객체 여러 개 리스트 |
| `result.scalar_one()` | 정확히 1개 (0개·2개면 에러) |
| `result.scalar_one_or_none()` | 0개면 `None`, 1개면 그거 |

진짜 배운 건 `selectinload(Robot.logs)` 쪽이다. echo 로그를 보면 SELECT가 **딱 두 번** 나간다.

```
SELECT ... FROM robots                            -- 1방: 로봇들 먼저
SELECT ... FROM task_logs WHERE robot_id IN (?, ?, ...)  -- 2방: 그 로봇들의 로그를 한 번에
```

robot이 100마리여도 SELECT는 2방으로 끝난다. 만약 이걸 안 쓰고 robot마다 `robot.logs`에 접근하면, 로봇 하나당 SELECT가 한 번씩 → **N+1 쿼리 폭탄**이 된다. 실무 성능 문제의 단골이다. `selectinload`(별도 SELECT로 IN 조회)와 `joinedload`(JOIN으로 한 방에) 둘 다 eager loading이고, 1:N에는 보통 `selectinload`가 무난하다.

## 함정 ① — async에서 lazy loading은 터진다 (`MissingGreenlet`)

일부러 `selectinload`를 빼고 `select(Robot)`만으로 조회한 뒤 `robot.logs`에 접근해봤다.

```
sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called;
can't call await_only() here. Was IO attempted in an unexpected place?
```

이게 오늘의 하이라이트. 무슨 일이 일어난 거냐면:

`selectinload`를 안 쓰면 `robot.logs`는 아직 DB에서 안 읽어온 상태로 남는다. 그러다 `robot.logs`에 접근하는 **바로 그 순간**, SQLAlchemy가 "로그가 필요하네? 지금 `SELECT ... FROM task_logs` 한 방 날려야지" 하고 **몰래 DB I/O를 시도**한다. 이게 lazy loading(지연 로딩)이다.

문제는 그 시점이 `await`도 없는 평범한 속성 접근(`robot.logs`)이라는 것. 동기 코드처럼 보이는 자리에서 비동기 I/O를 하려니까 async 엔진이 막는다. 에러 메시지의 `"IO attempted in an unexpected place"`가 정확히 그 뜻이다 — **예상 못 한 자리에서 DB 호출이 튀어나왔다.**

동기 SQLAlchemy였다면 이 lazy loading이 조용히 잘 동작한다(SELECT 한 번 더 나가고 끝). 그래서 **동기에서 멀쩡하던 코드를 async로 옮기면 이 에러를 만나는 게 통과의례**다.

| | 동기 SQLAlchemy | async SQLAlchemy |
|---|---|---|
| lazy loading (속성 접근 시 자동 SELECT) | 그냥 됨 | `MissingGreenlet` 💥 |
| 해결 | 신경 안 써도 됨 | `selectinload`/`joinedload`로 **미리** 로딩 |

> 여담: 설치할 때도 greenlet 때문에 한 번 막혔다. async SQLAlchemy는 동기 DBAPI 호출을 `greenlet`으로 감싸 `await` 지점으로 넘기는 구조라 `greenlet` 패키지가 필수인데, 내 환경에선 자동으로 안 깔려서 `uv add greenlet`을 따로 해줘야 했다(`sqlalchemy[asyncio]` extra로 받았으면 같이 끌려왔을 수도 있다 — 정확한 원인은 확실치 않다). lazy loading 에러도 결국 "그 greenlet 컨텍스트 밖에서 I/O를 시도해서" 나는 거라, 설치 에러와 런타임 에러가 같은 뿌리였다.
{: .prompt-info }

**교훈: async ORM에선 관계 데이터를 항상 eager로 미리 로딩하라.**

## 함정 ② — commit 후 객체가 조용히 죽는다 (`expire_on_commit`)

함정 ②에서 설명하겠다고 미뤄둔 `expire_on_commit=False`. 이걸 **기본값(`True`)**으로 두고 commit 직후 객체 속성에 접근하면 무슨 일이 나는지 확인했다.

```python
await session.commit()
print(r1.name)   # expire_on_commit=True(기본)면 여기서 사고
```

`expire_on_commit`의 기본값은 **`True`**다. 동기 SQLAlchemy의 철학은 "commit하면 DB가 진실, 메모리 객체는 못 믿는다"여서, commit 순간 세션에 매달린 **모든 객체의 속성을 만료(expired) 표시**한다. 그러면 다음에 `r1.name`에 접근할 때 **DB에서 최신값을 다시 읽어오는 SELECT**가 나간다.

문제는 — 그 재조회가 또 **lazy I/O**라는 것. 함정 ①과 정확히 같은 메커니즘으로 async에선 `MissingGreenlet`이 난다. commit은 분명 성공했는데, 그 직후 객체를 읽으려는 순간 죽는다.

그래서 **async + ORM에선 `expire_on_commit=False`가 사실상 표준 관용구**다. "commit 후에도 객체 속성을 메모리에 그대로 두고, 재조회하지 마라"는 뜻. 동기에선 안전 기본값이던 게 async에선 함정이 되는, 함정 ①과 같은 뿌리(=몰래 나가는 lazy I/O)의 다른 얼굴이다.

## 오늘의 정리

**SQLAlchemy 2.0 async** — `create_async_engine`(+`echo=True`로 SQL 관찰) → `Mapped`/`mapped_column` 타입 기반 모델 → `relationship`(cascade로 부모 add 시 자식 자동 저장) → `select()` + `result.scalars().all()` → `selectinload`로 N+1 회피.

그리고 async 고유의 함정 둘, 둘 다 **"몰래 나가는 lazy I/O"** 한 뿌리:
- **lazy loading** → `MissingGreenlet`. 해결: 관계는 `selectinload`/`joinedload`로 미리.
- **`expire_on_commit=True`(기본)** → commit 후 속성 접근이 재조회를 유발 → 역시 터짐. 해결: `expire_on_commit=False`.

지난 글에서 다진 "동기 vs async" 감각이 그대로 이어졌다. AsyncIO에서 `time.sleep`이 이벤트 루프를 멈추던 장면과, 여기서 lazy loading이 `await` 없이 I/O를 시도하다 막히는 장면은 사실 같은 이야기다 — **async 세계에서는 모든 I/O가 정직하게 `await`를 거쳐야 한다.** ORM이 그걸 등 뒤에서 몰래 하려 하면 그 자리에서 잡힌다.

다음은 **FastAPI**. 오늘까지 따로 배운 Pydantic(요청/응답 모델) + AsyncIO(async 엔드포인트) + SQLAlchemy(async ORM)가 드디어 하나의 동작하는 API로 합쳐진다. `expire_on_commit=False`로 만든 세션을 `Depends`로 주입하는 부분에서 오늘 배운 게 바로 다시 등장할 예정이다.
