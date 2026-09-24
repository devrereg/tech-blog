---
title: "[마인드리퍼블릭] FE/BE 통합 Monorepo 및 AI-Native 개발환경 구축"
date: 2026-09-07 04:00:00 +0900
categories: [포트폴리오]
tags: [monorepo, git-subtree, claude-code, ai-native, context-engineering]
description: "마인드리퍼블릭 WENOA의 FE/BE 저장소를 Git Subtree로 Monorepo에 통합해 기능 단위 개발 체계를 만들고, 폴더별 CLAUDE.md 기반 컨텍스트 엔지니어링으로 Claude Code의 토큰 부족 문제를 해결한 과정을 정리"
---


## 프로젝트 개요

- **회사 / 서비스**: 마인드리퍼블릭 / WENOA
- **프로젝트명**: FE/BE 통합 Monorepo 및 AI-Native 개발환경 구축
- **일정**: 2025.12 (3주)
- **역할**: Monorepo 구조 설계 · 저장소 통합 · AI 개발환경 구축
- **기술 스택**: Git Monorepo · Git Subtree · Claude Code · Claude Skills · CLAUDE.md


---

## 문제

FE와 BE가 별도 repository로 운영되어, AI를 활용해도 FE의 AI에게는 BE 스펙을, BE의 AI에게는 FE 구조를 따로 알려줘야 했다.

서비스 초기라 요구사항 변경이 잦았고, 변경될 때마다 FE·BE 개발자가 요청·응답 형식을 다시 맞추고 QA도 다시 해야 했다.

**AI를 쓰고 있어도, 저장소가 분리된 구조 때문에 생산성이 오르지 않았다.**

통합 이후에는 코드베이스가 커지면서 관련 없는 코드까지 컨텍스트에 적재되어 **토큰이 부족해지는 문제**가 생겼다.

---


## 해결

FE·BE를 **하나의 Monorepo로 통합**해 한 명의 개발자가 기능 단위로 FE·BE를 함께 개발하도록 바꾸고, 폴더별 컨텍스트 문서로 AI가 필요한 범위만 파악하도록 구성함.


**설계 기준**

- 기존 repository의 커밋 이력을 보존해야 함
- 한 명의 개발자가 하나의 브랜치에서 기능 전체를 개발·테스트할 수 있어야 함
- AI가 전체 코드베이스 중 필요한 범위만 파악할 수 있어야 함


## 아키텍처

![FE/BE 통합 AI-Native 개발환경 아키텍처(2025.12) 다이어그램](/assets/img/posts/mindrepublic-monorepo/wenoa_ai_native_dev_environment.png)


- **Monorepo 통합**: `git subtree add/pull --prefix=frontend|backend`로 기존 저장소를 이력째 가져와 `frontend/`, `backend/`로 구성
- **기능 단위 개발**: 하나의 feature 브랜치에서 FE·BE를 함께 개발하고, AI가 양쪽 코드를 참조해 E2E 테스트 케이스까지 작성
- **컨텍스트 엔지니어링**: 폴더별 `CLAUDE.md`와 파일 최상단에 역할·코드 컨벤션을 정의해, AI가 작업에 필요한 폴더와 파일만 선택적으로 탐색

**Git Subtree를 선택한 이유**: Submodule은 원본을 참조만 해 별도 checkout과 버전 동기화가 필요하지만, Subtree는 이력을 유지한 채 코드를 직접 포함해 하나의 저장소처럼 다룰 수 있다.

**CLAUDE.md를 폴더별로 나눈 이유**: 루트 하나에 모든 규칙을 담으면 매번 전체가 컨텍스트에 적재된다. 폴더 단위로 나누면 작업 대상 문서만 참조해 토큰을 줄이면서 정확도를 유지할 수 있다.

---

## 결과

- 1개월 예상 작업을 1주로 단축 (**약 75% 공수 절감**)
- AI 기반 E2E 테스트 케이스 작성으로 **테스트 약 80% 자동화**
- FE/BE 동기화 커뮤니케이션 비용 감소
- 컨텍스트 엔지니어링으로 토큰 부족 문제 해소

---

## 회고
- 개발자의 업무 비중이 코드 작성에서 기획 검토와 QA로 옮겨갔다.
- AI-Native 환경이라도 코드 리뷰를 통해 검증 하고, 가드레일링 하면서 고도화 해나가는 과정이 필요하다.