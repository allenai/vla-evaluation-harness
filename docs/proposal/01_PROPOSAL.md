# vla-evaluation-harness

## Problem

VLA (Vision-Language-Action) 연구에서 벤치마크 평가는 필수적이지만, 현실은 다음과 같다:

- 주요 벤치마크(LIBERO, CALVIN, SIMPLER, ManiSkill2 등)마다 환경 세팅이 전부 다르고, 의존성이 서로 충돌한다.
- 벤치마크마다 모델 연동 인터페이스가 다르다. 새 모델을 평가하려면 벤치마크 수만큼 integration 코드를 작성해야 한다.
- 이를 통합해주는 오픈소스가 사실상 없다.
- 기존 벤치마크는 전부 synchronous — agent가 추론하는 동안 환경 시간이 정지한다. 실제 로봇 배포 환경과 근본적으로 다르다.

## Goal

**다양한 로봇 벤치마크에서 VLA 모델을 단일 인터페이스로 평가할 수 있는 프레임워크.**

- 벤치마크 추가가 쉽다.
- 모델 추가가 쉽다.
- Real-time evaluation을 새로운 평가 축으로 제안한다.

## Non-Goal

- 모델 학습, 데이터 처리, 모델 내부 구현은 범위 밖이다.
- 단, model server의 인터페이스는 본 프레임워크가 엄격히 정의하며, 대표 모델들의 예시 server 구현을 제공한다.
- Action chunking은 원칙적으로 env의 책임이 아니다. 단, 이 패턴이 VLA 모델에서 매우 흔하게 사용되므로, model server 구현을 위한 관련 '도우미' 구현을 프레임워크가 제공한다.

## Architecture

### Client-Server 분리

모델 추론(server)과 벤치마크 시뮬레이션(client)을 완전히 분리한다.

- **Benchmark Client**: 시뮬레이션 환경 구동, observation 생성, action 적용, 메트릭 수집.
- **Model Server**: observation을 받아 action을 생성.

새 모델을 평가하려면 server만 교체. 새 벤치마크를 추가하려면 client adapter만 작성.

### 비동기 통신

Client-server 간 통신은 비동기 callback 기반으로 설계한다. 이는 real-time evaluation에서 환경 시간과 agent 추론 시간이 독립적으로 흘러야 하기 때문이다.

### 환경 격리

각 벤치마크의 시뮬레이션 환경을 격리하여 벤치마크 간 의존성 충돌 문제를 근본적으로 해결한다.

상세 설계는 [Architecture](../architecture.md) 및 [RFCs](../rfcs/README.md) 참조.

## Real-Time Evaluation

현재 모든 sim 벤치마크는 synchronous이다 — agent가 추론하는 동안 환경 시간이 정지한다. 실제 로봇에서는 추론이 느리면 로봇은 멈추거나 마지막 명령을 반복하고, 그 사이에 환경은 변한다. 현재 벤치마크는 이 현실을 반영하지 못한다.

Agent 실행과 독립적으로 env 시간이 흐르는 **real-time evaluation mode**를 제안한다. 기존 sync 평가와 병행하여, 추론 속도가 태스크 성공률에 미치는 영향을 정량적으로 드러낸다.

기존 대표 모델들(CogACT, Pi0, OpenVLA 등)에 대해 sync vs real-time 성공률 비교를 제공한다. 추론 속도와 chunk size 간의 trade-off를 정량적으로 드러낸다.

## Supported Benchmarks

| 우선순위 | Benchmark | 환경 | 특징 |
|---------|-----------|------|------|
| **Must** | LIBERO | 탁상 조작 (10개 suite) | VLA 표준 벤치마크 |
| **Must** | CALVIN | 탁상 조작 (long-horizon chain) | Multi-step sequential task |
| **Must** | SIMPLER | Real2Sim (Bridge/Google Robot) | Sim-to-real transfer 평가 |
| **Should** | BEHAVIOR-1K | 대규모 가정환경 (1,000 activities) | |
| **Should** | VLABench | VLA 특화 벤치마크 | |
| **Should** | LIBERO-Pro | LIBERO 확장 (harder tasks) | 고난도 조작 태스크 |
| **May** | RoboTwin | Dual-arm manipulation | |
| **May** | RoboCasa | 가정환경 조작 | |
| **May** | RLBench | 다양한 조작 태스크 | |

## References

설계 시 참고하는 기존 프로젝트:

| 프로젝트 | 참고 포인트 |
|----------|------------|
| [starVLA](https://github.com/starVLA/starVLA) | 다양한 벤치마크 통합 구현. 벤치마크 어댑터 구현 참고. |
| [dexbotic-benchmark](https://github.com/dexmal/dexbotic-benchmark) + [dexbotic](https://github.com/dexmal/dexbotic) | Client-server 분리 구조, Docker 기반 환경 격리, 벤치마크 어댑터 구현 참고. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI) | LLM 평가 프레임워크의 사실상 표준. 프레임워크 완성도, 사용성, 확장성의 기준점. |
| [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) ([eval docs](https://behavior.stanford.edu/challenge/evaluation.html#running-evaluations)) | WebSocket 기반 client/server 평가 설계. 비동기 통신 패턴 참고. |
| [X-VLA](https://github.com/2toinf/X-VLA) | 고품질 VLA 모델 연구. 벤치마크 eval 코드 포함 — 벤치마크 연동 구현 참고. |
| [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1) | 고품질 VLA 모델 연구. 벤치마크 eval 코드 포함 — 벤치마크 연동 구현 참고. |
| [RLinf](https://github.com/RLinf/RLinf) | RL 학습 인프라. LIBERO, CALVIN, ManiSkill, BEHAVIOR 등 다수 환경 wrapper 보유. 환경 통합 설계 참고. |
| [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (Meta) | Agentic RL용 통합 환경 프레임워크. Gymnasium-style 단일 API로 28+ 환경 통합, WebSocket client-server 통신, Docker 격리, 환경 scaffolding CLI. 다양한 환경을 단일 인터페이스로 추상화하는 설계 패턴 참고. |

## Competitive Landscape

본 프로젝트와 동일한 성격(client/server 아키텍처의 범용 VLA 벤치마크 평가 프레임워크 + real-time eval)을 가진 기존 프로젝트는 **0개**다. 8개 벤치마크 키워드의 nC2/nC3/nC4 조합 34건 이상의 exhaustive GitHub Code Search로 검증했다. 상세 서베이 결과는 [02_COMPETITIVE_LANDSCAPE.md](./02_COMPETITIVE_LANDSCAPE.md) 참조.

## Summary

1. **통합 인터페이스**: 하나의 인터페이스로 모든 벤치마크 × 모든 모델 조합을 평가.
2. **Real-time eval**: Agent 속도가 평가에 반영되는 새로운 축 제안.
3. **환경 격리**: 벤치마크 간 의존성 충돌 없이 실행.
4. **시장 공백 확인**: 동일 목적의 기존 프로젝트가 없음을 전수 조사로 검증.

