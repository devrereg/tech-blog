---
title: "FastAPI, 따로 배운 셋을 하나의 API로 묶기 — 의존성 생명주기를 중심으로"
date: 2026-06-26 15:35:37 +0900
categories: [Backend, Python]
tags: [python, fastapi, dependency-injection, async, sqlalchemy]
description: "Pydantic·AsyncIO·SQLAlchemy를 FastAPI 하나로 묶으며 배운 것 — Depends+yield의 의존성 생명주기, response_model+from_attributes 직렬화, expire_on_commit 함정, lifespan을 직접 돌려보며 정리한 학습 로그."
---

Python 4종 세트(Pydantic, AsyncIO, SQLAlchemy, FastAPI)의 마지막, FastAPI를 끝냈다. 앞선 셋이 각자 배운 조각이었다면 FastAPI는 그걸 HTTP 한 겹으로 묶는 접착제다. Pydantic은 요청/응답 스키마가 되고, AsyncIO는 `async def` 엔드포인트가 되고, SQLAlchemy의 async 세션은 `Depends`로 주입된다. 백엔드 경력이 있어서 라우팅·REST·DI 개념 자체는 익숙했다. 그래서 기초는 빠르게 넘기고 FastAPI 고유의 멘탈 모델, 특히 처음에 가장 헷갈렸던 **의존성(Dependency) 생명주기**에 시간을 가장 많이 썼다.

실습은 지난 SQLAlchemy 글의 `Robot`/`TaskLog` 모델을 그대로 가져와 로봇 CRUD API로 노출했다. `POST /robots`, `GET /robots`, `GET /robots/{id}`, `DELETE /robots/{id}` — 404 처리까지.

## 0. 요청 모델 ≠ 응답 모델

가장 먼저 길들인 습관. 클라이언트가 보내는 것과 서버가 돌려주는 것을 다른 스키마로 분리한다.

```python
from pydantic import BaseModel, ConfigDict

class RobotCreate(BaseModel):     # 클라이언트가 POST로 보내는 것
    name: str
    model: str

class RobotRead(BaseModel):       # 서버가 돌려주는 것
    model_config = ConfigDict(from_attributes=True)
    id: int                       # ← DB가 채워주는 PK. 입력엔 없고 출력엔 있다
    name: str
    model: str
```

왜 굳이 나누나. 처음엔 "필드 겹치는데 하나로 합치면 안 되나" 싶었다. 안 된다. 두 가지 이유다.

- **보안**: 요청/응답을 한 모델로 합치면 클라이언트가 `id`, `created_at`, 나아가 `is_admin` 같은 서버가 정해야 할 필드를 POST 바디로 주입할 수 있다(mass assignment). 반대로 응답엔 내부 필드가 과다 노출된다. 입력 스키마에 애초에 그 필드가 없으면 이 문제가 원천 차단된다.
- **계약 명확성**: 생성 시점엔 아직 `id`가 없다. 합쳐두면 "이 `id`를 클라가 보내야 하나?"가 `/docs`(OpenAPI)에서 모호해진다. 나누면 "입력은 name/model, 출력은 거기에 id 추가"가 문서에 그대로 계약으로 박힌다.

`model_config = ConfigDict(from_attributes=True)`는 잠시 뒤 2번에서 다시 나온다.

## 1. 의존성 생명주기 — Depends + yield의 진짜 의미

여기가 오늘의 핵심이고, 솔직히 처음에 잘못 이해하고 있던 부분이다. async 세션을 주입하는 코드는 이렇게 생겼다.

```python
from typing import Annotated
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine("sqlite+aiosqlite:///./robot.db", echo=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():
    async with async_session() as session:
        yield session            # ← 여기가 핵심

SessionDep = Annotated[AsyncSession, Depends(get_session)]
```

그리고 엔드포인트에서는 이렇게 받는다.

```python
@app.post("/robots", response_model=RobotRead, status_code=201)
async def create_robot(payload: RobotCreate, session: SessionDep):
    ...
```

`Depends`는 한마디로 DI 컨테이너다. Spring의 `@Autowired`, NestJS의 provider와 같은 자리. 요청마다 `get_session`을 호출해 세션을 만들어 주입하고, 요청이 끝나면 정리한다. `SessionDep`처럼 `Annotated[타입, Depends(...)]`를 타입 별칭으로 묶어두면 모든 엔드포인트에서 `session: SessionDep` 한 줄로 재사용된다.

내가 틀렸던 지점이 바로 `yield`의 타이밍이었다. "yield 위/아래 코드가 다 요청 받을 때 같이 실행된다"고 막연히 생각했는데, 완전히 틀렸다. 정확한 모델은 이렇다.

```text
       요청 도착
          │
   ┌──────▼───────┐
   │ yield 이전   │  ← 세션 생성 (async with 진입). 핸들러 진입 "전"
   └──────┬───────┘
          │  yield session  → 핸들러에 세션을 넘겨줌
   ┌──────▼───────┐
   │  핸들러 실행 │  ← create_robot / get_robot ... 응답 생성
   └──────┬───────┘
          │  (핸들러 종료, 또는 예외 발생)
   ┌──────▼───────┐
   │ yield 이후   │  ← async with 블록 종료 → 세션 close/rollback. 응답 "후"
   └──────┬───────┘
          │
       응답 반환
```

즉 `yield`를 기준으로 시간이 갈린다. yield 이전은 요청 처리 시작 시점, yield 이후는 응답이 끝난 뒤 정리 시점. 둘은 동시에 실행되는 게 아니라 핸들러를 사이에 두고 앞뒤로 나뉜다.

이게 왜 강력하냐면, FastAPI의 yield 의존성은 `try`/`finally`처럼 동작하기 때문이다. 백엔드에서 커넥션 반납할 때 쓰던 그 패턴 그대로다.

```python
# get_session의 async with는 개념적으로 이것과 같다
async def get_session():
    session = async_session()
    try:
        yield session          # 핸들러가 여기서 세션을 받아 쓴다
    finally:
        await session.close()  # 핸들러가 끝나거나 "예외가 나도" 반드시 실행
```

그래서 핸들러 안에서 `raise HTTPException(404)`가 터져도 — finally에 해당하는 yield 이후 정리(close/rollback)는 반드시 실행된다. 세션이 새지 않는다. 내가 손으로 `try`/`finally`를 쓰지 않아도 `async with` + `yield` 조합이 그 보장을 대신 해준다.

정리: yield 이전 = 요청 시작 시 셋업, yield 이후 = 응답 종료 시(예외 포함) 정리. `Depends`가 이 둘을 핸들러 앞뒤에 자동으로 끼워 넣는다.

## 2. ORM 객체를 그냥 return 했는데 JSON이 나온다 — 두 개의 설정

CRUD 핸들러는 이렇게 생겼다. SQLAlchemy 실습에서 배운 `session.get`/`select`가 그대로 등장한다.

```python
@app.post("/robots", response_model=RobotRead, status_code=201)
async def create_robot(payload: RobotCreate, session: SessionDep):
    robot = Robot(name=payload.name, model=payload.model)
    session.add(robot)
    await session.commit()
    return robot                       # ← ORM 객체를 그냥 반환?!

@app.get("/robots/{robot_id}", response_model=RobotRead)
async def get_robot(robot_id: int, session: SessionDep):
    robot = await session.get(Robot, robot_id)   # PK 단건 조회는 session.get
    if robot is None:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")
    return robot

@app.get("/robots", response_model=list[RobotRead])
async def list_robots(session: SessionDep):
    result = await session.execute(select(Robot))
    return result.scalars().all()
```

`return robot`은 Pydantic 모델이 아니라 SQLAlchemy ORM 객체다. 그런데도 클라이언트엔 깔끔한 JSON이 나간다. 여기엔 두 개의 설정이 얽혀 있는데, 처음에 나는 둘의 역할을 헷갈렸다(그리고 한쪽은 실제로 돌려보니 생각과 달랐다).

- **라우트 데코레이터의 `response_model=RobotRead`** — "이 핸들러의 반환값을 `RobotRead`로 변환·검증해서 내보내라." 응답 직렬화의 책임자이자 `/docs` 스키마의 출처. 반환값에 `RobotRead`에 없는 필드가 있어도 잘려나가서 과다 노출도 막힌다.
- **스키마의 `from_attributes=True`** — Pydantic은 기본적으로 dict를 받는데, 우리가 주는 건 `robot.id`처럼 속성으로 접근하는 객체다. 이 옵션이 "dict 말고 객체의 속성을 읽어서 채워라"를 켜준다. (Pydantic v1의 `orm_mode`가 v2에서 이름이 바뀐 것.) [FastAPI 공식 문서](https://fastapi.tiangolo.com/tutorial/sql-databases/)도 ORM 객체를 반환할 땐 이 옵션을 켜라고 안내한다.

`from_attributes=True`를 지우면? — 여기서 처음 생각과 달랐다. 직접 두 경로로 돌려봤다.

```python
# (A) 내 손으로 검증을 부르면 → 터진다
RobotRead.model_validate(robot)   # from_attributes 없음
# pydantic_core._pydantic_core.ValidationError: 1 validation error for RobotRead
#   id  Input should be a valid dictionary or instance of RobotRead [type=model_type, ...]

# (B) 그런데 FastAPI의 response_model 직렬화 경로로는 → 그냥 201로 나간다 (FastAPI 0.138 + Pydantic 2.13에서 직접 확인)
```

즉 `from_attributes`는 **검증**(`model_validate`로 객체→모델을 *만들* 때) 스위치이지, 응답 **직렬화**에까지 강제되는 건 아니다. 현재 FastAPI의 직렬화 경로는 모델 필드 이름으로 객체 속성을 그냥 읽어내서, 이 옵션이 없어도 응답은 나간다. 그래도 켜두는 게 맞다 — 공식 문서 권장이고, 명시적 `model_validate`·구버전·SQLModel 관계 로딩에선 여전히 필요하다. "두 설정이 짝을 이룬다"기보다 "`response_model`이 직렬화를 책임지고, `from_attributes`는 객체 기반 검증을 위한 보험"으로 정리하는 게 정확하다.

여기서 헷갈리지 말아야 할 것: **직렬화 책임**(`response_model` + `from_attributes`)과 다음 장의 **세션 만료**(`expire_on_commit`)는 완전히 다른 레이어의 이야기다. 이름이 비슷해 섞기 쉬운데, 하나는 "객체→JSON", 하나는 "commit 후 객체 상태"다.

`POST`엔 `status_code=201`, `DELETE`엔 `status_code=204`(본문 없음)를 명시한다. 처음에 `status=201`로 썼다가 한 번 막혔는데, 에러 메시지가 친절하게 파라미터 이름을 알려준다 — 직접 재현한 출력은 이렇다.

```text
TypeError: FastAPI.post() got an unexpected keyword argument 'status'
```

파라미터 이름은 `status`가 아니라 `status_code`다.

## 3. 함정 — expire_on_commit, SQLAlchemy 때 깔아둔 그 한 줄이 여기서 터진다

지난 SQLAlchemy 글에서 `async_sessionmaker(..., expire_on_commit=False)`를 "async에선 거의 항상 끈다"고만 하고 넘어갔다. FastAPI에서 왜 그게 결정적인지가 오늘 드러났다. 일부러 기본값(`True`)으로 돌려놓고 POST를 날려봤다.

```python
async_session = async_sessionmaker(engine, expire_on_commit=True)   # 일부러 고장
```

클라이언트는 **HTTP 500**을 받고, 서버 로그엔 `fastapi.exceptions.ResponseValidationError`가 찍힌다. 핵심은 그 에러가 가리키는 위치다 — 직접 재현한 출력을 보면:

```text
fastapi.exceptions.ResponseValidationError: 1 validation error
  {'type': 'get_attribute_error',
   'loc': ('response', 'id'),          # ← 바로 robot.id 추출 지점
   'msg': "Error extracting attribute: MissingGreenlet: greenlet_spawn has not "
          "been called; can't call await_only() here. Was IO attempted in an "
          "unexpected place?"}
```

`loc`가 `('response', 'id')`인 게 결정적이다 — `commit()`은 통과했는데, 바로 다음 `response_model`이 응답을 만들려고 `robot.id`를 읽는 그 순간에 `MissingGreenlet`이 터진 것이다. 메커니즘을 3줄로 정리하면:

1. `commit()`이 끝나면 SQLAlchemy는 세션의 모든 객체를 expire(만료) 시킨다 — "이 속성값은 이제 못 믿어, DB가 진실".
2. `response_model=RobotRead`가 직렬화하려고 `robot.id`에 접근하는 순간, 만료된 속성이라 **DB를 다시 조회(lazy load)**하려 한다 = 새 I/O 발생.
3. 그런데 그 접근은 더 이상 `await`로 감싼 async 컨텍스트가 아니라 동기 자리에서 I/O를 시도하는 꼴 → async 엔진이 막는다 → `MissingGreenlet`.

지난 글의 lazy loading 함정과 정확히 같은 뿌리다 — "몰래 나가는 lazy I/O". 그때는 `robot.logs` 접근에서, 이번엔 commit 직후 `robot.id` 접근에서 같은 일이 터진 것뿐이다. `expire_on_commit=False`는 "commit 후에도 메모리 값을 그대로 믿고 재조회하지 마라"라서 이 lazy load 자체를 막는다.

| 상황 | `expire_on_commit=True` (기본) | `expire_on_commit=False` |
| --- | --- | --- |
| commit 직후 `robot.id` 접근 | 재조회 시도 → async에선 `MissingGreenlet` 💥 | 메모리 값 그대로 사용, 안전 |
| FastAPI에서 commit 후 `return robot` | 터진다 | 잘 돌아간다 |

피하는 다른 방법도 있다. `await session.refresh(robot)`로 명시적으로 다시 불러오면 되지만, 매 핸들러마다 쓰기 번거롭다. 그래서 async + FastAPI에선 `expire_on_commit=False`를 세션 팩토리에 기본으로 까는 게 관용구다.

## 4. 테이블 생성은 lifespan에 — 그리고 프로덕션에선 쓰면 안 되는 이유

처음에 POST를 날리자마자 `no such table: robots`로 500이 났다. 어디서도 테이블을 안 만들었기 때문. SQLAlchemy 실습에선 `main()` 안에서 `create_all`을 했지만, FastAPI에선 앱이 뜰 때 한 번 실행해야 한다. 그 자리가 `lifespan`이다.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)   # startup: 테이블 생성
    yield                                                # 이 위가 시작 시, 아래가 종료 시
    # (shutdown 정리 코드가 있다면 yield 아래)

app = FastAPI(lifespan=lifespan)
```

`lifespan`은 서버 startup/shutdown 훅이다. Spring의 `@PostConstruct`, NestJS의 `onModuleInit`과 같은 자리. 여기 둔 이유는 명확하다 — 앱당 정확히 1회, 시작 시점에 보장된다. import 시점이나 라우터 함수 안에 두면 테스트·멀티프로세스 환경에서 부작용이 난다.

재밌는 건 `lifespan` 자체도 `yield`로 startup/shutdown을 가른다는 것. 1번의 `get_session`과 똑같은 패턴이다 — yield 위는 시작, 아래는 끝. FastAPI 곳곳에서 이 "yield로 앞뒤를 가르는" 멘탈 모델이 반복된다.

한 가지 분명히: `create_all`은 프로덕션용이 아니다. 흔한 오해가 "매번 테이블을 새로 만들어 위험하다"인데, 사실 `create_all`은 이미 있으면 새로 만들지 않는다(드롭/재생성이 아님). 진짜 문제는 스키마 변경(컬럼 추가·타입 변경)을 반영하지 못한다는 것. 모델에 컬럼을 추가해도 기존 테이블은 그대로다. 그래서 실무에선 Alembic 같은 마이그레이션 도구로 스키마를 버전 관리한다. `create_all`은 토이/로컬 부트스트랩 전용으로만 쓴다.

## 오늘의 정리

FastAPI — 따로 배운 Pydantic·AsyncIO·SQLAlchemy가 드디어 하나의 동작하는 async CRUD API로 합쳐졌다.

1. **요청/응답 모델 분리**(`RobotCreate`/`RobotRead`) — 보안(mass assignment 차단)과 계약 명확성.
2. **의존성 생명주기** — `Depends` + `async with ... yield`. yield 이전=요청 시작 셋업, 이후=응답 종료 정리(`try`/`finally`처럼 예외에도 보장). 오늘 가장 크게 바로잡은 멘탈 모델.
3. **`response_model` + `from_attributes`** — ORM 객체를 return해도 JSON이 나가는 직렬화 레이어. 직렬화를 책임지는 건 `response_model`이고, `from_attributes`는 객체 기반 검증(`model_validate`)을 위한 보험 — 현재 FastAPI 직렬화 경로엔 후자가 없어도 응답이 나간다는 걸 직접 확인했다.
4. **`expire_on_commit=False`** — commit 후 lazy 재조회를 막아 `MissingGreenlet`을 회피. 세션 상태 레이어. (직렬화와 헷갈리지 말 것.)
5. **`lifespan`** — startup에서 `create_all`. 단, 프로덕션은 Alembic.

관통하는 주제는 두 개였다. 하나는 지난 글부터 이어진 "async에선 모든 I/O가 정직하게 `await`를 거쳐야 한다" — `expire_on_commit` 함정이 그 세 번째 변주였다. 다른 하나는 FastAPI 특유의 **yield로 앞뒤를 가르는 생명주기 모델** — `get_session`에서도, `lifespan`에서도 같은 모양으로 반복됐다. 이걸 한 번 제대로 잡으니 의존성, 미들웨어, 백그라운드 작업까지 같은 틀로 보이기 시작했다.

이걸로 Python 4종 세트 완주. 다음은 1단계 AI 파트, LangChain으로 넘어간다. 오늘 만든 CRUD 골격(async 엔드포인트 + `Depends` 주입 + 요청/응답 모델)은 앞으로 나올 PDF RAG 서비스, AI 모델 서빙 API에 그대로 재사용되는 뼈대다.
