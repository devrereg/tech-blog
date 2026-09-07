---
title: "[마인드리퍼블릭] FE/BE 통합 AI-Native 개발환경 구축"
date: 2026-09-07 04:00:00 +0900
categories: [포트폴리오]
tags: [portfolio, mindrepublic, wenoa, monorepo, git-subtree, claude-code, ai-native, context-engineering, developer-productivity]
description: "마인드리퍼블릭 WENOA의 FE/BE 통합 개발 환경 구축. 별도로 운영하던 프론트엔드·백엔드 저장소를 Git Subtree로 하나의 Monorepo에 합쳐 기능 단위 개발이 가능하도록 만들고, 폴더별 CLAUDE.md 기반 컨텍스트 엔지니어링으로 Claude Code의 토큰 부족 문제를 해결했다."
---

> 마인드리퍼블릭에서 운영하는 서비스 **WENOA**의 **FE/BE 통합 AI-Native 개발 환경** 구축 기록입니다. 별도로 운영하던 프론트엔드·백엔드 저장소를 Git Subtree로 하나의 Monorepo에 합치고, Claude Code가 전체 코드베이스를 효율적으로 다루도록 컨텍스트 구조를 설계했습니다.

## 프로젝트 개요

- **회사 / 서비스**: 마인드리퍼블릭 / WENOA
- **프로젝트명**: FE/BE 통합 AI-Native 개발환경 구축
- **일정**: 2025.12 (3주)
- **기술 스택**: Git Mono-repo · Git Subtree · Claude Code (Claude Agent) · Claude Skills · Harness Architecture

**설계 기준**

- FE·BE를 하나의 저장소로 통합하되, 기존 repository의 커밋 이력을 보존
- AI Agent가 필요한 범위만 골라 파악할 수 있도록 코드베이스의 컨텍스트 구조를 명시적으로 설계

---

## 아키텍처

![FE/BE 통합 AI-Native 개발환경 아키텍처(2025.12) 다이어그램. 기술 스택은 Git Mono-repo, Git Subtree, Claude Agent, Claude Skills. 통합 전에는 FE Repository와 BE Repository가 분리돼 있고, 두 저장소 사이를 수동 동기화(커뮤니케이션 비용 증가)로 맞춘다. FE 개발자와 BE 개발자가 각각 git subtree add/pull --prefix=frontend, --prefix=backend 명령으로 코드를 Monorepo(WENOA)로 가져온다. Monorepo 안에는 frontend/(구 FE Repo)와 backend/(구 BE Repo) 디렉터리가 있고, 그 아래 Feature 단위 브랜치에서 FE와 BE를 동시에 수정해 1개의 PR로 처리한다. 오른쪽의 Claude Agent(Claude Code)는 FE/BE 전체 코드베이스에 접근해 Monorepo를 대상으로 코드 분석과 생성을 수행한다](/assets/img/posts/mindrepublic-monorepo/wenoa_ai_native_dev_environment.png)

*FE/BE 통합 AI-Native 개발 환경 (2025.12) — 분리 운영하던 FE·BE 저장소를 Git Subtree로 Monorepo(WENOA)에 통합하고, feature 단위로 FE·BE를 함께 수정해 1개 PR로 처리한다. Claude Code는 통합된 전체 코드베이스를 참조해 작업한다.*

---

## 문제

### FE·BE 분리 운영으로 인한 개발 생산성 저하

- FE와 BE가 **별도 repository**로 운영됐다. 백엔드에서 API·데이터 구조를 변경하면, 그 변경을 프론트엔드와 맞추는 동기화 과정이 매번 필요했다.
- 이 동기화가 사람 간 커뮤니케이션으로 이뤄져, 기능 하나를 끝내는 데 FE·BE 조율 비용이 반복적으로 들었고 개발 속도의 병목으로 작용했다.

### Monorepo 통합 이후의 토큰 부족

- FE·BE를 한 저장소로 합치자 AI가 참조해야 할 컨텍스트 크기가 커졌다.
- Claude Code로 작업할 때 관련 없는 코드까지 컨텍스트에 적재되면서 **토큰이 부족해지는 현상**이 발생했다.

---

## 해결

### Monorepo 통합 + AI-Native 개발 환경 구축

- FE·BE 각 마이크로서비스를 **하나의 Monorepo(WENOA)로 통합**했다. 기존 FE/BE repository를 각각 `frontend/`, `backend/` 디렉터리로 두고, `git subtree add/pull --prefix=frontend|backend`로 **원본 저장소의 커밋 이력을 유지**하며 가져왔다.
- 통합 이후에는 하나의 feature 브랜치에서 FE·BE를 동시에 수정하고 **1개의 PR로 묶어** 처리한다. 역할 경계 대신 기능 단위로 개발이 흐르도록 만들었다.
- Claude Code 기반 **AI-Native 개발 환경**을 구성해, AI Agent가 FE·BE 전체 코드베이스를 한 번에 참조하며 코드를 분석·생성하도록 했다.

### 컨텍스트 엔지니어링으로 토큰 문제 해결

- 프로젝트 루트에 **Monorepo 구조를 명시적으로 정의**했다.
- 폴더별 `CLAUDE.md`를 두고, 각 파일 최상단에 해당 폴더의 **역할과 코드 컨벤션**을 기술했다. AI가 작업에 필요한 폴더만 선택적으로 파악하도록 유도해, 불필요한 컨텍스트 적재를 줄였다.

---

## 결과

### FE·BE 분리 운영으로 인한 개발 생산성 저하

- FE/BE 역할 구분 없이 **기능(feature) 단위 개발**이 가능해졌다.
- 기존에 1개월이 걸릴 것으로 예상한 작업을 **1주로 단축**했다(약 75% 공수 절감).

### Monorepo 통합 이후의 토큰 부족

- 컨텍스트 엔지니어링으로 **토큰 부족 문제를 해소**했다.
- 동시에 AI가 코드베이스를 효율적으로 파악하는 **AI-Native 개발 환경**을 갖췄다.
