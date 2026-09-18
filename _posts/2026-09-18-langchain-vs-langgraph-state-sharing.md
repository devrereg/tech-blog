---
title: "LangChain 대신 LangGraph를 선택한 이유: State 공유 관점에서"
date: 2026-09-18 17:30:00 +0900
categories: [Backend, Python]
tags: [langchain, langgraph, ai-agent, llm, agent-framework, state-management, python]
description: "AgentExecutor도 이미 tool-calling 루프를 돈다. LangChain과 LangGraph의 진짜 차이는 루프 유무가 아니라, 노드 간에 타입이 있는 State를 공유할 수 있는가였다."
---

> [WENOA AX 서비스]({% post_url 2026-09-07-mindrepublic-wenoa-ax-influencer-search %})의 자연어 인플루언서 검색 Agent는 LangGraph로 만들었습니다. 그런데 막상 지금 구현된 그래프는 `agent`와 `tools` 두 노드가 순환하는 구조뿐이고, 이 정도는 LangChain의 `AgentExecutor`로도 충분히 만들 수 있습니다. 그렇다면 왜 LangGraph를 선택했을까요? "루프를 돌 수 있어서"라는 흔한(그리고 부정확한) 답 대신, 실제 코드 레벨에서 두 프레임워크가 어떻게 다른지 비교하며 진짜 이유를 정리합니다.

## TL;DR

- `AgentExecutor`도 내부적으로 tool-calling 루프를 돕니다. "루프가 필요해서 LangGraph"는 근거가 약합니다.
- 진짜 차이는 **State를 공유하는가**입니다. LangChain은 단계 간 데이터를 텍스트(메시지)로 넘기고 LLM이 다시 해석해야 하지만, LangGraph는 타입이 있는 State를 모든 노드가 직접 읽고 씁니다.
- 이 차이는 멀티턴 대화에서 더 뚜렷해집니다. LangChain의 대화 메모리는 최종 답변 텍스트만 남기고, tool 호출 디테일은 매 턴 증발합니다.
- 기능(검색, 즐겨찾기, 캠페인 관리 등)이 하나둘 늘어날수록 이 차이는 설계 전체를 흔드는 문제가 됩니다.

## 1. LCEL: 두 프레임워크가 공유하는 기반

먼저 짚어야 할 건, LangGraph가 LangChain과 완전히 별개의 라이브러리가 아니라는 점입니다. LangChain의 기본 합성 단위는 `Runnable`이고, `|` 연산자로 이어 붙이는 LCEL(LangChain Expression Language)로 체인을 만듭니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘")
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# prompt, llm, parser는 각각 Runnable이고,
# | 로 이어 붙이면 그 자체로 새로운 Runnable(체인)이 된다
chain = prompt | llm | parser

chain.invoke({"topic": "LangGraph"})   # 동기 단건 실행
chain.batch([...])                     # 여러 입력 병렬 처리
chain.stream({"topic": "LangGraph"})   # 토큰 단위 스트리밍
```

`RunnableBranch`로 조건 분기를, `RunnableParallel`로 병렬 실행을, `RunnablePassthrough.assign`으로 데이터 누적을 표현할 수 있습니다.

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    # 입력 dict를 유지한 채 "context"라는 새 키만 추가
    RunnablePassthrough.assign(context=lambda x: retrieve(x["query"]))
    | ChatPromptTemplate.from_template("컨텍스트: {context}\n질문: {query}")
    | llm
    | parser
)
```

여기까지 보면 LangChain도 데이터를 다음 단계로 넘기는 수단이 있어 보입니다. 하지만 이건 **정해진 순서를 한 방향으로만 흐르는 파이프라인**입니다. 노드를 정의하고 조건에 따라 되돌아가는 사이클을 만드는 개념은 없습니다. `AgentExecutor`는 바로 이 LCEL 위에 "tool-calling 루프"라는 특정 패턴 하나를 완제품으로 얹어놓은 것입니다.

## 2. AgentExecutor는 이미 루프를 돈다

```python
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def search_youtube_creators(categories: list[int], subscriber_min: int) -> str:
    """구독자 수와 카테고리로 유튜브 크리에이터를 검색한다."""
    return f"[검색 결과] categories={categories}, subscriber_min={subscriber_min}"

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 크리에이터 검색을 도와주는 에이전트야."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # 이전 tool 호출/결과가 텍스트로 주입되는 자리
])

agent = create_tool_calling_agent(llm, [search_youtube_creators], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_youtube_creators], verbose=True)

agent_executor.invoke({"input": "구독자 10만 이상 뷰티 유튜버 찾아줘"})
```

`verbose=True`로 실행하면 다음과 같은 로그가 찍힙니다.

```text
> Entering new AgentExecutor chain...

[1번째 루프] LLM 판단: search_youtube_creators 호출
  Invoking: `search_youtube_creators` with `{'categories': [1], 'subscriber_min': 100000}`
  [검색 결과] categories=[1], subscriber_min=100000

[2번째 루프] LLM 판단: tool_calls 없음 → 결과 요약 후 종료
  "조건에 맞는 뷰티 유튜버를 찾았어요: ..."

> Finished chain.
```

`[LLM 판단] ↔ [tool 실행]`을 반복하는 이 구조는, LangGraph의 `agent ↔ tools` 사이클과 패턴상 동일합니다. 즉 **"루프가 필요해서 LangGraph를 썼다"는 설명은 정확하지 않습니다.** LangChain도 이미 루프를 돕니다.

## 3. 진짜 차이: State 공유 여부

`AgentExecutor`가 루프를 도는 동안 쌓이는 기록은 `intermediate_steps`입니다.

```python
result = agent_executor.invoke(
    {"input": "구독자 10만 이상 뷰티 유튜버 찾아줘"},
    return_intermediate_steps=True,
)

for action, observation in result["intermediate_steps"]:
    print(action.tool, action.tool_input, "->", observation)
# search_youtube_creators {'categories': [1], ...} -> [검색 결과] ...
```

문제는 이 `intermediate_steps`가 **딱 그 `.invoke()` 호출 하나의 범위 안에서만** 존재한다는 점입니다. 함수가 끝나면 사라지고, 다음 단계나 다음 기능으로 정보를 넘길 방법이 없습니다. 결국 다음 단계가 이전 결과를 활용하려면 최종 응답을 텍스트로 남기고, 그 텍스트를 LLM이 다시 읽고 해석하는 수밖에 없습니다. 이 재해석 과정에서 정보 손실이나 왜곡이 생길 수 있습니다.

물론 "`intermediate_steps`를 별도 저장소에 직접 쌓아서 다음 `invoke()` 때 프롬프트에 다시 넣어주면 되지 않나?"라는 반론이 가능합니다. 하지만 그렇게 해도 결국 `agent_scratchpad`나 `chat_history` 같은 메시지 자리에 텍스트로 밀어 넣는 것이라, LLM이 그걸 다시 파싱해야 한다는 본질은 그대로입니다. 구조화된 필드를 코드에서 직접 읽고 쓰는 것과, 텍스트를 LLM에게 다시 해석시키는 것은 다른 문제입니다.

LangGraph는 이 문제를 State로 풉니다.

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # add_messages 리듀서가 없으면 messages는 노드가 반환할 때마다 덮어써진다.
    # 리스트를 "누적"하려면 반드시 Annotated로 리듀서를 지정해야 한다.
    messages: Annotated[list, add_messages]
    campaign_id: str | None
    selected_creators: list

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)   # LLM 호출: 구현 생략
workflow.add_node("tools", tool_node)    # tool 실행: 구현 생략

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    lambda x: "tools" if x["messages"][-1].tool_calls else END,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
```

`agent`와 `tools` 노드는 서로 다른 역할을 하지만, 같은 `AgentState`를 공유합니다. 검색 결과를 텍스트로 남기고 다시 해석하는 대신, `selected_creators` 같은 구조화된 필드에 직접 쓰고 읽을 수 있습니다.

## 4. 멀티턴 대화에서 더 벌어지는 차이

"메모리를 붙이면 LangChain도 문제없지 않나?"라는 질문이 자연스럽게 따라옵니다. 확인해보면 그렇지 않습니다.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_memory.invoke(
    {"input": "구독자 10만 이상 뷰티 유튜버 찾아줘"},
    config={"configurable": {"session_id": "user-1"}},
)

# store["user-1"]에 쌓이는 건 딱 이것뿐:
#   [HumanMessage("구독자 10만 이상 뷰티 유튜버 찾아줘"),
#    AIMessage("조건에 맞는 뷰티 유튜버를 찾았어요: ...")]
#
# 실제로 어떤 파라미터(categories, subscriber_min 등)로 검색했는지는
# intermediate_steps에만 있었고, 메모리 저장 대상이 아니라서 사라진다.
```

`RunnableWithMessageHistory`가 저장하는 건 `HumanMessage`/`AIMessage`, 즉 최종 텍스트뿐입니다. 예를 들어 다음 턴에 "그중에서 구독자 더 많은 사람만 다시 보여줘"라고 물으면:

```python
agent_with_memory.invoke(
    {"input": "그중에서 구독자 더 많은 사람만 다시 보여줘"},
    config={"configurable": {"session_id": "user-1"}},
)
```

LLM에게 넘어가는 `chat_history`에는 `subscriber_min=100000`이라는 실제 검색 조건이 없습니다. 앞선 답변 텍스트("조건에 맞는 뷰티 유튜버를 찾았어요: ...")에 그 숫자가 그대로 노출돼 있었다면 LLM이 다시 파싱해 복원할 수도 있지만, 요약 과정에서 "구독자 10만 이상"이라는 표현이 빠졌거나 다른 말로 바뀌었다면 복원을 보장할 수 없습니다. 검색 조건이 필드로 남아있지 않고 자연어 요약 한 번을 더 거쳐야 한다는 것 자체가 구조적인 손실 지점입니다.

LangGraph의 `MemorySaver` 체크포인터는 `thread_id` 단위로 **State 전체**를 저장·복원합니다. `messages`뿐 아니라 `selected_creators`, `campaign_id`처럼 직접 정의한 필드까지 다음 턴에 그대로 이어받을 수 있습니다.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-1"}}
app_graph.invoke({"messages": [...]}, config=config)

# 다음 턴: 같은 thread_id로 다시 invoke하면
# messages뿐 아니라 selected_creators, campaign_id까지 그대로 복원된다
app_graph.invoke({"messages": [...]}, config=config)
```

## 5. 기능이 늘어날수록 벌어지는 격차

WENOA Agent에는 크리에이터 검색 외에도 즐겨찾기, 캠페인 관리처럼 앞으로 붙을 기능이 더 있습니다. `AgentExecutor`는 tool-calling을 위한 고정된 루프 패턴 하나만 지원하기 때문에, 서로 다른 역할의 기능을 추가하려면 그 루프 바깥에서 오케스트레이션 로직을 매번 새로 짜야 합니다.

```python
# LangGraph: 노드와 엣지만 추가하면 됨
workflow.add_node("favorite", favorite_node)
workflow.add_node("campaign", campaign_node)

workflow.add_conditional_edges(
    "agent",
    route_by_intent,
    {"tools": "tools", "favorite": "favorite", "campaign": "campaign", END: END},
)
# 기존 agent <-> tools 루프 구조는 건드리지 않고 확장된다
```

기존 `agent ↔ tools` 루프를 재설계하지 않고도, 같은 State를 공유하는 노드를 옆에 추가하기만 하면 됩니다.

## 정리

| 관점 | LangChain (AgentExecutor) | LangGraph |
|---|---|---|
| 루프 | 있음 (tool-calling 전용 고정 패턴) | 있음 (노드/엣지로 자유롭게 설계) |
| 단계 간 데이터 전달 | 텍스트(메시지)로 남기고 LLM이 재해석 | 타입이 있는 State를 노드가 직접 공유 |
| 실행 범위 | `intermediate_steps`는 한 번의 `invoke()` 안에서만 유효 | State는 체크포인터로 턴을 넘어 유지 |
| 멀티턴 메모리 | 최종 메시지만 저장, tool 호출 디테일은 유실 | State 전체가 스냅샷으로 보존 |
| 기능 확장 | 루프 밖에서 오케스트레이션 직접 구현 | 노드/엣지 추가로 확장 |

지금 당장은 `agent ↔ tools` 두 노드뿐이라 이 장점이 크게 체감되지 않을 수 있습니다. 하지만 다음 단계로 즐겨찾기 노드와 캠페인 노드를 추가할 계획이고, 그때는 지금 정의한 `AgentState`에 필드 몇 개와 노드 몇 개만 얹으면 됩니다. `AgentExecutor`였다면 그 시점에 가서 오케스트레이션 구조 자체를 다시 설계해야 했을 겁니다. 즉 이번 선택은 지금 당장의 요구보다, 다음에 붙일 기능을 겨냥한 결정에 가깝습니다.
