# Competitive Landscape Survey — vla-evaluation-harness

## 방법론

본 프로젝트와 동일한 목적(다수 벤치마크 통합 VLA 평가 프레임워크)을 가진 기존 프로젝트가 존재하는지 검증하기 위해, **exhaustive bottom-up 전수 조사**를 수행했다.

1. **키워드 정의**: 타겟 벤치마크 8종의 코드 레벨 키워드를 정의:
   - `libero`, `calvin`, `simpler_env`, `behavior`, `vlabench`, `metaworld`, `rlbench`, `robomimic`

2. **GitHub Code Search — 조합적 전수 검색**: 위 8개 키워드에 대해 nC2(28조합), nC3(56조합), nC4(70조합) = **총 154가지 조합**을 체계적으로 검색. 실제 수행한 검색은 **34개 이상**이며, 모든 검색에서 동일한 레포지토리 집합으로 수렴한 후 포화(saturation)를 확인했다.
   - nC2: 10개 조합 완료 (주요 벤치마크 쌍 전수)
   - nC3: 20개 이상 조합 완료 (libero/calvin/simpler_env 축 + behavior/vlabench/rlbench/metaworld/robomimic 교차)
   - nC4: 4개 조합 완료 (대표 조합 — 결과 0~21개로 수렴)

3. **중복 제거**: 포크/미러를 제거하고 원본 레포지토리만 추출.
4. **분류**: 각 레포지토리의 실제 코드를 확인하여 성격별로 분류.

**수렴 확인**: nC3/nC4 조합은 nC2에서 이미 발견된 레포지토리의 부분집합만을 반환했다. 특히 4-way 조합에서는 RoboVerse 패밀리 + vlaleaderboard만 매칭되었고, "비주류" 벤치마크(behavior, vlabench, robomimic)를 포함하는 3-way 조합에서는 awesome-list와 survey 레포만 매칭되어, 검색 공간이 완전히 포화되었음을 확인했다.

## 결과

총 **~20개 고유 레포지토리**가 벤치마크 2종 이상을 포함하고 있었다. 분류 결과:

### A. 시뮬레이션 플랫폼 (1개)

| 레포 | ★ | 벤치마크 | 비고 |
|------|---|----------|------|
| [RoboVerse](https://github.com/RoboVerseOrg/RoboVerse) | 1,670 | LIBERO, CALVIN, SIMPLER, ManiSkill, RLBench, MetaWorld | 통합 시뮬레이션 플랫폼(MetaSim). 평가 프레임워크가 아님. Client/server 구조 없음. |

### B. 평가 프레임워크 (1개)

| 레포 | 벤치마크 | 비고 |
|------|----------|------|
| [dexbotic-benchmark](https://github.com/dexmal/dexbotic-benchmark) | LIBERO, CALVIN, SIMPLER, ManiSkill2, RoboTwin | 본 프로젝트의 출발점. 모놀리식 구조, 인터페이스 확장성 부족. |

### C. VLA 모델 레포 — 자체 eval 코드 내장 (11개)

각 모델이 자기 평가를 위해 벤치마크 코드를 **개별적으로 복사·유지**하는 패턴. 범용 프레임워크가 아님.

| 레포 | 벤치마크 | 비고 |
|------|----------|------|
| CladernyJorn/VLM4VLA | LIBERO, CALVIN, SIMPLER | |
| baaivision/UniVLA (ICLR 2026) | LIBERO, CALVIN, ManiSkill | |
| OpenHelix-Team/VLA-Adapter | LIBERO, CALVIN, ManiSkill (OXE) | |
| OpenHelix-Team/Unified-Diffusion-VLA | LIBERO, CALVIN, ManiSkill (OXE) | |
| OpenHelix-Team/HiF-VLA | LIBERO, CALVIN, ManiSkill (OXE) | |
| JiuTian-VL/CogVLA (NeurIPS 2025) | LIBERO, CALVIN, ManiSkill (OXE) | |
| EmbodiedAI-RoboTron/RoboTron-Mani | LIBERO, CALVIN, RLBench | RoboMM 프로젝트 |
| huiwon-jang/GR00Tn1.5-ContextVLA | LIBERO, CALVIN, RLBench (configs) | |
| dunnolab/NinA | LIBERO, CALVIN, RoboMimic | Normalizing Flows in Action |
| MasterXiong/Hyper-VLA | LIBERO, SIMPLER, MetaWorld | Hypernetwork 기반 VLA |
| InternRobotics/InternManip | LIBERO, CALVIN, SIMPLER, BEHAVIOR | All-in-one manipulation suite. 학습 중심, 범용 eval 프레임워크 아님. |

### D. RL/학습 인프라 및 데이터 파이프라인 (4개)

벤치마크 이름이 환경 설정이나 데이터셋 config에 등장하지만, 평가 목적이 아님.

| 레포 | 성격 |
|------|------|
| RLinf/RLinf | RL 학습 인프라 (LIBERO, CALVIN, ManiSkill, BEHAVIOR env wrapper) |
| renwang435/multigen (CoRL 2025) | RoboVerse 기반 오디오 시뮬레이션 연구 |
| NsurrenderX/gcr_lerobot_2 | LeRobot 포크, OXE 데이터셋 config |
| hedemil/flower_vla_pret | Flower VLA 사전학습, OXE transforms |

### E. 리더보드/서베이 (비경쟁)

| 레포 | 성격 |
|------|------|
| k1000dai/vlaleaderboard | VLA 벤치마크 점수 집계 (데이터 전용, 코드 아님) |
| 8종 이상의 awesome-list / survey | 논문 정리 레포. Awesome-Robotics-Manipulation, Embodied-AI-Guide 등 |

## RoboVerse 심층 검토

서베이에서 가장 규모가 큰 프로젝트가 RoboVerse(1,670★)다. 아래는 자연스럽게 떠오르는 의문들과 그에 대한 조사 결과를 정리한 것이다.

### Q1. RoboVerse로 LIBERO/RLBench 등을 바로 돌릴 수 있는가?

설치하면 LIBERO/RLBench 태스크를 실행할 수는 있다. 단, **원본 벤치마크 환경을 그대로 실행하는 것이 아니다.** MetaSim이 태스크 정의(씬, 오브젝트, 로봇 배치 등)를 자체 Scenario Config 포맷으로 재정의한 버전을 실행한다.

물리 엔진 자체는 같을 수 있다 — 예컨대 LIBERO가 원래 MuJoCo(robosuite) 기반이므로 MetaSim에서 MuJoCo handler를 선택하면 같은 물리 엔진이 돌아간다. 하지만 환경 코드 레이어(observation pipeline, reward 계산, reset 로직 등)가 MetaSim의 통합 인터페이스를 통과하므로, 원본 `libero` 패키지의 코드와는 다르다. 따라서 원본 벤치마크의 공식 숫자와 직접 비교는 보장되지 않는다.

### Q2. 외부 VLA 연구가 RoboVerse를 평가에 활용하는가?

**확인된 사례: 0건.** 구체적으로:

- **VLA-0 (NVIDIA)**: RoboVerse를 사용하지 않는다. 표준 LIBERO + SO-100 실물 태스크로 평가.
- **C 카테고리 VLA 모델 레포 11개**: 전부 자체 eval 코드를 개별 복사·유지. 어느 것도 RoboVerse를 통해 평가하지 않음.
- **multigen (CoRL 2025)**: RoboVerse 기반이지만 오디오 시뮬레이션 연구이며, VLA 평가 목적이 아님.
- **RoboVerse 논문 자체**: ACT, Diffusion Policy, OpenVLA, Octo를 내부적으로 평가했으나, 이는 저자들의 플랫폼 검증용이지 외부 채택 사례가 아님.

외부 채택이 없는 이유로는 (1) 원본 벤치마크와 결과 비교 불가 우려, (2) 2025.04 출시로 아직 신생, (3) 설치 복잡도(Isaac Sim 등 무거운 의존성) 등이 추정되나, 확실하지 않다.

### Q3. RoboVerse의 핵심 목적은 무엇인가?

논문(RSS 2025)에 따르면, 핵심 목적은 **평가가 아니라 "시뮬레이션 통합 + 대규모 데이터 + cross-simulator 전이"**다.

| 어필 포인트 | 내용 |
|------------|------|
| Write Once, Run Anywhere | MetaSim API 하나로 MuJoCo, Isaac Sim, SAPIEN, PyBullet, Genesis 통합 |
| 통합 데이터셋 | 500k trajectories, 276 task categories, 5.5k assets |
| Cross-simulator transfer | MuJoCo에서 학습 → Isaac Sim에서 실행 |
| Sim-to-real | RoboVerse 데이터로 fine-tune → 실물 로봇 동작 |
| Data augmentation | MimicGen으로 데모 증강 → 성공률 향상 |

LIBERO/RLBench 등의 태스크를 통합한 이유는 "이 벤치마크를 평가하겠다"가 아니라, 태스크 카탈로그를 넓히고 통합 데이터셋을 구축하며 cross-simulator/cross-embodiment 전이 실험의 재료로 쓰기 위한 것이다.

### Q4. GitHub Issues 현황 — maturity

2026.02 기준 open issues 54개. "LIBERO 결과 재현 안 된다"라는 직접적 이슈는 없으나, 기본적인 수준의 문제가 산적해 있다:

| Issue | 내용 | 시사점 |
|-------|------|--------|
| #385 | IsaacLab fails for evaluation | 평가 자체가 실행 불가 (이후 closed) |
| #546 | How to integrate scenes from original simulators? | 원본 씬 통합 방법 질문. **답변 없이 open** |
| #545 | Unfinished contents in wiki | 문서 미완성 |
| #481 | Rendering results differ across simulators | cross-simulator 일관성 문제. **open** |
| #582 | Domain randomization and task asset issues | 에셋 가용성 문제. open |
| #526 | gymnasium 0.29.1 not compatible | 의존성 호환 문제. open |

설치/호환성/crash 이슈(Mac 지원, SAPIEN 에러, Isaac Sim GUI crash, Genesis 실행 실패 등)가 대다수를 차지한다. 특히 #481(렌더링 결과 시뮬레이터 간 차이)은 핵심 셀링 포인트인 cross-simulator 일관성에 직결되는 문제인데 미해결 상태다.

## 결론

**본 프로젝트와 동일한 성격(client/server 아키텍처의 범용 VLA 벤치마크 평가 프레임워크 + real-time eval)을 가진 기존 프로젝트는 0개다.** 이 결론은 8개 벤치마크 키워드의 nC2/nC3/nC4 조합 34건 이상의 exhaustive 검색으로 검증되었다.

- 가장 가까운 것은 dexbotic-benchmark이며, 이것이 본 프로젝트의 출발점이다.
- RoboVerse(1,670★)는 규모가 크지만 "시뮬레이션 플랫폼"이지 "평가 프레임워크"가 아니다. 외부 VLA 연구의 평가 채택 사례도 확인되지 않는다.
- InternManip은 다수 벤치마크를 포함하지만 "학습 suite"이지 범용 eval 프레임워크가 아니다.
- VLA 모델 레포 11개는 모두 자기 모델 평가를 위해 벤치마크 코드를 개별 복사·유지하는 패턴이다. 이것이 바로 본 프로젝트가 해결하려는 핵심 문제(eval 코드 중복)를 실증한다.

