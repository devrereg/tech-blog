---
title: "[마인드리퍼블릭] 정기결제 기능 개발을 위한 토스페이먼츠 연동"
date: 2026-09-07 03:00:00 +0900
categories: [포트폴리오]
tags: [portfolio, mindrepublic, wenoa, tosspayments, payment, recurring-billing, spring-boot, kotlin, mysql, slack-bot, compensating-transaction]
description: "마인드리퍼블릭 WENOA의 정기결제 기능 개발을 위한 토스페이먼츠 연동. TossPayments API와 내부 MySQL 간 트랜잭션 정합성을 확보하고, 결제 성공 후 DB 적재 실패에 대비한 보상 트랜잭션(자동 환불)과 Slack 실시간 알림 구조를 구현했다."
---

> 마인드리퍼블릭에서 운영하는 서비스 **WENOA**의 **정기결제 기능** 개발 기록입니다. 토스페이먼츠(TossPayments) API를 연동해 빌링키 기반 정기결제를 붙이면서, **외부 PG 결제 결과와 내부 DB 상태가 어긋나는 상황**을 어떻게 감지하고 되돌리는지에 초점을 맞췄습니다.

## 프로젝트 개요

- **회사 / 서비스**: 마인드리퍼블릭 / WENOA
- **프로젝트명**: 정기결제 기능 개발을 위한 토스페이먼츠 연동
- **일정**: 2026.01 (1개월)
- **기술 스택**: Spring Boot 3 (Kotlin) · TossPayments API · MySQL · Slack Bot

**설계 기준**

- TossPayments API와 내부 DB 간 트랜잭션 정합성 확보
- 결제 실패 로그 적재 및 재처리(보상 트랜잭션) 구조 구현

---

## 아키텍처

![정기결제 토스페이먼츠 연동 아키텍처(2026.01) 다이어그램. 기술 스택은 Spring Boot3(Kotlin), TossPayments API, MySQL, Slack Bot. 사용자가 API 서버(Spring Boot3 + Kotlin)로 정기결제(빌링키) 등록을 요청하면, API 서버는 TossPayments API(카드 등록·결제)로 카드 등록/결제를 요청하고 success 응답을 받는다. success 수신 후 API 서버는 MySQL에 결제 정보 저장을 시도(트랜잭션)한다. 이 저장 과정에서 TossPayments v1(레거시) API의 response 스펙 변경으로 예외가 발생한다. DB 저장이 성공하면 결제 정보 등록을 완료하고 사용자에게 응답한다. 예외가 발생하면(No) 보상 트랜잭션을 실행해 TossPayments API로 자동 환불을 요청하고 환불을 완료하며, 결제 실패 로그를 MySQL failure_log에 적재하고, Slack Bot으로 실패 사유와 Trace ID를 담은 알림을 전송해 Slack 채널의 CS 담당자가 실시간으로 대응한다](/assets/img/posts/mindrepublic-tosspayments/wenoa_tosspayments_architecture.drawio.png)

*정기결제 토스페이먼츠 연동 흐름 (2026.01) — TossPayments에서 카드 등록·결제가 성공(success)하면 서버가 결제 정보를 MySQL에 등록한다. 이 DB 적재 단계에서 예외가 나면 보상 트랜잭션으로 TossPayments 결제를 자동 환불하고, 실패 로그(failure_log)를 적재하며, Slack Bot으로 CS 담당자에게 실시간 통지한다.*

---

## 문제

정기결제 연동 과정에서 **외부 PG(TossPayments)와 내부 DB의 상태가 어긋나는** 상황이 발생했다.

- TossPayments에서는 **카드 등록과 결제가 정상적으로 성공**했으나, 서버가 그 결과(빌링키·결제 승인 정보)를 **DB에 저장하는 과정에서 예외가 발생**했다.
- 원인은 **TossPayments v1(레거시) API의 response 스펙 변경**으로 파악됐다. 응답 필드 구조가 바뀌면서 서버의 파싱/매핑 로직이 예외를 던졌다.
- 결과적으로 **고객은 결제됐는데 서비스에는 결제 기록이 없는** 정합성 깨짐 상태가 만들어졌고, 이를 자동으로 감지하거나 되돌릴 장치가 없었다.

---

## 해결

DB 적재 단계에서 예외가 발생하면 **이미 승인된 TossPayments 결제를 자동으로 되돌리고**, 그 사실을 실시간으로 알리는 구조를 구현했다.

- **보상 트랜잭션(자동 환불)** — 결제 정보 등록 트랜잭션이 실패하면, 직전에 성공한 TossPayments 결제 건을 **자동 환불(취소) API로 롤백**한다. 외부 결제와 내부 DB가 "둘 다 성공" 또는 "둘 다 없음" 중 하나로만 수렴하도록 만들어 정합성을 확보했다.
- **실패 로그 적재** — 실패한 결제 건의 요청/응답 원문, 예외 내용, 환불 처리 결과를 별도 테이블에 남겨 **재처리와 원인 분석의 근거**로 삼았다.
- **Slack 실시간 알림** — 결제 실패와 자동 환불 처리 결과를 Slack Bot으로 즉시 통지해, **CS 팀이 대시보드를 보지 않아도 장애를 인지**하고 고객 대응에 바로 착수할 수 있게 했다.
- **재발 방지** — v1(레거시) API의 응답 스펙 변경에 흔들리지 않도록 응답 파싱/매핑 로직을 방어적으로 보강했다.

---

## 결과

- DB 적재 실패 시 **TossPayments 결제가 자동 환불**되어, "고객은 결제됐는데 기록이 없는" 정합성 깨짐 상태가 자동으로 해소된다.
- 결제 실패 상황을 **Slack으로 즉시 인지**하게 되어 신속한 CS 대응 체계를 마련했다.
- 실패 로그를 남기면서 원인(v1 API 응답 스펙 변경)을 특정했고, **재발 방지를 위한 결제 로직 보강**을 진행했다.
