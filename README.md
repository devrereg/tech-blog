# devrereg's tech blog

[![Build and Deploy](https://github.com/devrereg/tech-blog/actions/workflows/pages-deploy.yml/badge.svg)](https://github.com/devrereg/tech-blog/actions/workflows/pages-deploy.yml)
[![GitHub license](https://img.shields.io/github/license/devrereg/tech-blog.svg?color=blue)][mit]

> 백엔드 개발을 공부하며 정리한 기록을 모아두는 기술 블로그입니다.

**🔗 [devrereg.github.io/tech-blog](https://devrereg.github.io/tech-blog)**

## About

어떤 요구사항이 와도 안정적으로 시스템을 구현할 수 있는 개발자가 되는 것을 목표로, 공부한 내용을 아카이빙하는 공간입니다. [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 테마 기반 Jekyll 블로그로 GitHub Actions를 통해 GitHub Pages에 자동 배포됩니다.

**관심 분야:** 대규모 트래픽의 안정적 운영 · 동시성 제어 · 시스템 아키텍처

## 최근 글

<!-- BLOG-POST-LIST:START -->
<!-- BLOG-POST-LIST:END -->

## 주요 시리즈

- **쿠폰 발급 동시성 제어** — Lost Update 재현부터 비관적 락, 낙관적 락, Redis 분산락, Kafka 비동기 처리까지 하나의 문제를 단계적으로 해결해나가는 6부작 시리즈
- **Spring Boot** — AOP, Flyway/DDL 자동화 등 실무에서 부딪힌 이슈 정리
- **PyTorch / Deep Learning** — 선형 회귀, 이진 분류 등 딥러닝 기초 실습 기록

## 기술 스택

| 영역 | 사용 기술 |
| --- | --- |
| 정적 사이트 생성 | Jekyll, Chirpy Theme |
| 배포 | GitHub Actions → GitHub Pages |
| 주요 주제 | Spring Boot, Kotlin, Concurrency, Python, PyTorch |

## 로컬에서 실행하기

```shell
bundle install
bundle exec jekyll s
```

## 연락

- GitHub: [@devrereg](https://github.com/devrereg)
- Email: devrereg@gmail.com

---

이 저장소는 [Chirpy Starter][chirpy]를 기반으로 하며, [MIT][mit] 라이선스를 따릅니다.

[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[mit]: https://github.com/devrereg/tech-blog/blob/main/LICENSE
