---
title: "[마인드리퍼블릭] WENOA AX 서비스 — 자연어 기반 인플루언서 검색 AI Agent 개발"
date: 2026-09-07 02:00:00 +0900
categories: [포트폴리오]
tags: [ai-agent, llm, langgraph, nuxt, vue, typescript, fastapi, python, sse, ecs-fargate, serverless, github-actions]
description: "마인드리퍼블릭 WENOA의 내부 직원용 AX 서비스. 멀티 필터 방식의 인플루언서 검색을 자연어 검색으로 전환. NuxtJS 채팅 UI, SSE 스트리밍, FastAPI + LangGraph Agent, ECS Fargate 서버리스 방식의 설계/배포한 내용을 정리"
---


## 프로젝트 개요

- **회사 / 서비스**: 마인드리퍼블릭 / WENOA
- **프로젝트명**: WENOA 전용 AX 서비스 개발 (자연어 기반 인플루언서 검색)
- **일정**: 2026.03 (2주)
- **역할**: 1인 풀스택 개발 (채팅 UI · API · AI Agent · 인프라 · CI/CD)
- **기술 스택**
  - Frontend: NuxtJS (Vue) · TypeScript · SSE
  - Backend: FastAPI (Python) · LangGraph, 기존 API (Spring boot3 · kotlin, Elasticsearch)
  - Infra: AWS (ECS Fargate, ALB, ECR, Route 53, Secrets Manager, CloudWatch, IAM) · Docker · GitHub Actions


---

## 문제

기존 Elasticsearch 기반 인플루언서 검색은 **멀티 필터 방식**이었다. 이 방식은 사용자가 **검색어, 카테고리, 팔로워 수, 조회수 등** 적합한 검색 조건을 직접 골라 설정해야 했고, 검증하는 작업에 소요되는 시간이 많이 걸렸다.

---


## 해결

자연어 요청을 입력받아 **AI Agent가 검색 필터를 자동 생성**하고, 필터를 통해 기존 검색 API를 호출함으로써 유저가 검색 필터를 고민하는 데 소요되는 시간을 단축함.


**설계 기준**

- 필터를 몰라도 원하는 인플루언서를 찾을 수 있어야 함
- Agent 처리 중에도 사용자가 진행 상황을 실시간으로 확인할 수 있어야 함
- 기존 멀티 필터 검색 사용 방식에 영향을 주지 않아야 함
- 내부 직원 전용·저트래픽 서비스이므로 상시 운영 비용을 최소화해야 함
- Agent로서 기능 확장이 용이해야 함


## 아키텍처

![WENOA AX 서비스 아키텍처(2026.03) 다이어그램](/assets/img/posts/mindrepublic-wenoa-ax/wenoa_ax_architecture.png)


LangGraph로 설계한 Graph
![node 와 edge graph](/assets/img/posts/mindrepublic-wenoa-ax/wenoa_ax_langgraph.png)


- **agent 노드 (LLM)**: 자연어 요청을 해석해 검색어·카테고리·팔로워 수·조회수 등 검색 필터를 생성하고, tool_calls로 tools 노드에 전달
- **tools 노드**: 전달받은 필터로 기존 멀티 필터 검색 API를 호출하고 결과를 agent에 반환
- agent가 더 이상 tool_calls를 만들지 않으면 최종 응답과 함께 종료

필터 생성(판단)은 LLM이, 검색(실행)은 검증된 기존 API가 맡도록 역할을 분리했다.


**LangGraph를 선택한 이유**: LangChain으로 구현할 수도 있었지만, State 공유, 멀티턴 대화 대응, 기능 확장 시 재설계 비용을 고려해 LangGraph를 선택함.

**SSE를 선택한 이유**: Agent 진행 상황 전달은 "서버 → 클라이언트" 단방향이면 충분했고, SSE는 일반 HTTP 위에서 동작해 기존 ALB 구성을 그대로 쓸 수 있어 WebSocket보다 구현과 운영이 단순하기 때문이다.

---

## 결과

- 인플루언서 리스트업 소요 시간 **75% 단축** (기존 평균 약 1시간 → 15분)
- 검색 필터를 고민하고 검증하는 과정을 Agent에게 위임
- 저트래픽 내부 서비스를 ECS Fargate 서버리스로 구성해 상시 운영 비용 절감

---

## 회고
- 자연어 + 이미지 기반으로 멀티모달 방식의 요청을 처리할 수 있도록 개발했으면 사용성이 더 개선되었을 것 같다.
- 확장을 고려해 LangGraph로 설계했지만, 기능 범위가 단편적이라 graph 구조가 단순한 상태에서 마무리되었다. 확장한다면 인플루언서 마케팅에 필요한 다른 기능들(인플루언서 즐겨찾기, 제안서 메일로 보내기 등)을 담당하는 노드를 추가해 보고 싶다.
