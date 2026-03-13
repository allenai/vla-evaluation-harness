# dexbotic-benchmark + dexbotic — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | [Dexmal/dexbotic-benchmark](https://github.com/Dexmal/dexbotic-benchmark) + [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic) |
| 주요 목적 | VLA 모델(CogACT, π0, OFT 등)의 멀티-벤치마크(CALVIN, LIBERO, SimplerEnv, RoboTwin 2.0, ManiSkill2) 평가 프레임워크 |
| 우리 프로젝트에서의 참고 포인트 | Client-server 분리 아키텍처, Docker 기반 환경 격리, 벤치마크 어댑터 패턴 (01_PROPOSAL.md 명시) |

## 프로젝트 구조

두 개의 별도 저장소로 구성:

### dexbotic-benchmark (평가 클라이언트)
```
evaluation/
├── evaluator/
│   ├── base_evaluator.py          # BaseEvaluator ABC
│   ├── libero_evaluator.py        # LIBERO 평가 구현
│   ├── calvin_evaluator.py        # CALVIN 평가 구현
│   ├── simpler_evaluator.py       # SimplerEnv 평가 구현
│   ├── robotwin2_evaluator.py     # RoboTwin 2.0 평가 구현
│   └── maniskill2_evaluator.py    # ManiSkill2 평가 구현
├── policies/
│   ├── base_vla_agent.py          # BaseVLAAgent (HTTP 클라이언트 + 액션 청킹)
│   ├── adaptive_ensemble.py       # Temporal ensemble 구현
│   ├── libero_vla_agent.py        # LIBERO 전용 에이전트
│   ├── calvin_vla_agent.py        # CALVIN 전용 에이전트
│   ├── simpler_vla_agent.py       # SimplerEnv 전용 에이전트
│   ├── robotwin2_vla_agent.py     # RoboTwin 전용 에이전트
│   └── maniskill2_vla_agent.py    # ManiSkill2 전용 에이전트
├── configs/                       # 벤치마크별 YAML 설정
├── run_libero_evaluation.py       # LIBERO 진입점
├── run_calvin_evaluation.py       # CALVIN 진입점
├── run_simpler_evaluation.py      # SimplerEnv 진입점
├── run_robotwin2_evaluation.py    # RoboTwin 진입점
└── run_maniskill2_evaluation.py   # ManiSkill2 진입점
scripts/env_sh/                    # conda 환경 활성화 + 실행 쉘 스크립트
Dockerfile                         # 모든 벤치마크 환경을 하나의 컨테이너에 구축
```

### dexbotic (모델 서버 + 학습)
```
dexbotic/
├── exp/
│   ├── base_exp.py                # Flask 모델 서버 (InferenceConfig)
│   ├── cogact_exp.py              # CogACT 모델 서버
│   ├── pi0_exp.py / pi05_exp.py   # π0/π0.5 모델 서버
│   ├── oft_exp.py                 # OFT 모델 서버
│   └── ...                        # 기타 모델 서버들
├── client.py                      # DexClient (HTTP 요청 래퍼)
├── sim_envs/
│   ├── base.py                    # BaseEnvWrapper ABC (RL 학습용)
│   └── factory.py                 # 배치 환경 팩토리
├── model/                         # VLA 모델 아키텍처
└── data/                          # 학습 데이터 파이프라인
playground/benchmarks/             # 벤치마크별 모델 서버 런치 스크립트
```

## 핵심 분석

### 벤치마크 추상화

**BaseEvaluator ABC** (`dexbotic-benchmark/evaluation/evaluator/base_evaluator.py`):

```python
class BaseEvaluator(ABC):
    def __init__(self, config: OmegaConf, output_structure: Dict[str, Path]): ...
    @abstractmethod
    def setup_environment(self) -> None: ...
    @abstractmethod
    def setup_model(self) -> None: ...
    @abstractmethod
    def _run_evaluation_impl(self) -> Dict[str, Any]: ...
    def run_evaluation(self) -> Dict[str, Any]:
        self.setup_environment()
        self.setup_model()
        return self._run_evaluation_impl()
```

예를 들어, `LiberoEvaluator._run_evaluation_impl()`은 약 150줄에 걸쳐 환경 초기화, task 순회, 에피소드 루프, 성공률 계산, 비디오 저장을 모두 처리한다. `CalvinEvaluator`도 유사한 로직을 별도로 구현한다.

**우리 설계와의 비교**: 우리의 `Benchmark` ABC + `EpisodeRunner` 패턴은 "무엇(Benchmark)"과 "어떻게(EpisodeRunner)"를 분리하여 이 문제를 해결한다. dexbotic의 접근은 빠른 프로토타이핑에는 유리하지만 확장성에서 한계가 있다.

### 모델 통합 패턴

**BaseVLAAgent** (`dexbotic-benchmark/evaluation/policies/base_vla_agent.py`):

모델 서버와의 통신을 담당하는 에이전트 계층. 핵심 기능:

1. **HTTP 클라이언트**: `requests.post`로 `/process_frame` 엔드포인트에 요청
2. **액션 청킹 관리**: `replan_step` 파라미터로 버퍼에서 액션 소비 주기 제어
3. **Adaptive Ensemble**: 코사인 유사도 기반 가중 평균으로 겹치는 액션 청크 블렌딩

```python
class BaseVLAAgent:
    def predict(self, obs: dict, task_description: str) -> np.ndarray:
        if self.step_count % self.replan_step == 0:
            raw_actions = self._call_server(obs, task_description)
            self.action_buffer = self._process_actions(raw_actions)
        return self.action_buffer[self.step_count % self.replan_step]
```

각 벤치마크별 에이전트 서브클래스(LiberoVLAAgent, CalvinVLAAgent 등)는 **관측 전처리**(이미지 리사이즈, 크롭 등)만 오버라이드한다.

**서버 측** (`dexbotic/dexbotic/exp/base_exp.py`):
- `InferenceConfig` 데이터클래스가 Flask 서버를 호스팅
- 단일 엔드포인트: `POST /process_frame`
- 이미지는 multipart form data로 수신, 텍스트는 form field로 수신
- 응답: `{"response": [action_array]}` JSON
- `threaded=False` — 동시 요청 처리 불가

**평가**: 통신 프로토콜이 극도로 단순(동기 HTTP + JSON). 이미지를 PNG 인코딩하여 전송하므로 latency overhead가 크다. 우리 설계의 WebSocket + msgpack 바이너리 프로토콜과 비교하면 성능 차이가 클 것.

### 에피소드 실행 루프

에피소드 루프가 각 Evaluator 서브클래스에 하드코딩되어 있다. LIBERO 기준:

```python
# LiberoEvaluator._run_evaluation_impl() 내부 (의사코드)
for task_id, task in enumerate(tasks):
    env = create_env(task)
    for episode in range(num_episodes):
        obs = env.reset()
        for step in range(max_steps):
            action = agent.predict(obs, task.description)
            obs, reward, done, info = env.step(action)
            if done: break
        success = check_success(info)
        results.append(success)
    save_video(frames)
save_results_json(results)
```

**특징**:
- 동기적 step-by-step 실행만 지원
- 실시간(real-time) 평가 모드 없음
- 에피소드 타임아웃은 각 벤치마크가 정의한 `max_steps`로만 제어
- 벤치마크 간 루프 구조가 유사하지만 공통 추출되지 않음

### 통신 구조

| 측면 | 내용 |
|------|------|
| 프로토콜 | HTTP/1.1 REST (Flask) |
| 직렬화 | 이미지: PNG multipart, 텍스트: form data, 응답: JSON |
| 동시성 | `threaded=False` — 단일 요청씩 처리 |
| 포트 | 기본 7891 |
| 네트워크 | `--network host`로 Docker 컨테이너와 호스트 공유 |
| 연결 관리 | 매 프레임마다 새 HTTP 연결 (persistent connection 없음) |

**평가**: 가장 단순한 선택지. 프로토타이핑에는 충분하지만, 프레임당 HTTP 오버헤드 + PNG 인코딩/디코딩이 실시간 평가에는 부적합. 우리의 WebSocket + msgpack + 원시 바이트 이미지 전송 설계가 이를 개선한다.

### 환경 격리

**Docker 전략**: 단일 모놀리식 Dockerfile.

```dockerfile
# 핵심 구조 (의사코드)
FROM nvidia/cuda:12.1-devel-ubuntu22.04
# 1. 기본 시스템 패키지 설치
# 2. Miniforge(conda) 설치
# 3. 벤치마크별 conda 환경 생성:
#    - simpler_env (Python 3.10)
#    - calvin_env  (Python 3.10)
#    - libero_env  (Python 3.10)
#    - RoboTwin    (Python 3.10)
#    - maniskill2_env (Python 3.10)
# 4. 각 환경에 해당 벤치마크 dependencies 설치
# 5. GPU 렌더링 설정 (EGL, Vulkan)
```

**특징**:
- **하나의 거대 이미지에 5개 conda 환경** — 이미지 크기 매우 큼 (예상 20GB+)
- 실행 시 쉘 스크립트(`scripts/env_sh/*.sh`)로 적절한 conda 환경 활성화
- 모델 서버는 Docker 외부(호스트)에서 실행, `--network host`로 통신
- 벤치마크 추가 시 Dockerfile 전체 재빌드 필요

**평가**: 빠른 구현이지만 확장성이 낮다. 우리 설계의 "벤치마크별 독립 Docker 이미지" 접근이 이를 개선한다. 다만, 단일 이미지의 장점도 있다 — CI에서 하나만 빌드/테스트하면 됨.

### 설정/CLI

**설정 체계**: OmegaConf YAML, 벤치마크별 1개 파일.

```yaml
# evaluation/configs/libero/example_libero.yaml
evaluator:
  type: libero
  num_episodes: 20
  max_steps: 300
model_server:
  host: localhost
  port: 7891
  model_name: cogact
output_dir: ./result_test/libero
```

**CLI**: 통합 CLI 프레임워크 없음. 각 벤치마크가 독자적 진입점을 가짐:
- `run_libero_evaluation.py --config path/to/yaml`
- `run_calvin_evaluation.py --config path/to/yaml`
- ... 이하 동일 패턴

실행 흐름: `docker run ... bash scripts/env_sh/libero.sh config.yaml`
→ conda 환경 활성화 → `python run_libero_evaluation.py --config config.yaml`

**평가**: 설정은 적절히 단순하지만, 통합 진입점이 없어 사용자가 벤치마크별로 다른 명령어를 기억해야 한다. lm-evaluation-harness의 단일 `lm_eval` CLI와 대비된다.

### 결과 수집

- **출력 형식**: JSON 파일. 벤치마크별로 구조가 다름.
- **LIBERO**: `{task_name: {success_rate: float, episodes: [{success: bool, steps: int}]}}`
- **비디오 녹화**: `imageio`로 에피소드별 MP4 저장. 프레임을 메모리에 쌓아두고 에피소드 종료 시 기록.
- **통합 결과 뷰 없음**: 벤치마크 간 결과를 하나로 모아 보는 기능이 없다.

**평가**: 최소한의 결과 기록. 우리의 `ResultCollector`가 벤치마크 간 통합 결과 뷰를 제공하는 것은 차별점이 될 수 있다.

## 프로젝트별 심층 분석

### Adaptive Ensemble 구현

`adaptive_ensemble.py`의 temporal ensemble은 주목할 만하다:

```python
class AdaptiveEnsemble:
    def __init__(self, action_dim, max_chunk_size):
        self.buffer = []  # [(actions, weight)]

    def add_chunk(self, actions: np.ndarray):
        self.buffer.append(actions)

    def get_action(self) -> np.ndarray:
        # 코사인 유사도 기반 가중 평균
        weights = compute_cosine_weights(self.buffer)
        return weighted_average(self.buffer, weights)
```

단순한 exponential weighting 대비 코사인 유사도 기반 가중치는 액션 청크 간 불일치가 큰 경우 자동으로 오래된 청크의 가중치를 낮춘다. CogACT 논문에서 유래한 기법으로, 우리 설계의 `PredictModelServer.action_ensemble=callable` 옵션에 매핑된다. 이 코사인 유사도 기반 앙상블 로직을 callable 함수로 구현하여 `action_ensemble` 파라미터에 전달하면 된다.

### dexbotic 서버 측의 모델 다형성

`dexbotic/exp/` 디렉토리에는 모델별 서버 구현이 있다:
- `cogact_exp.py`, `pi0_exp.py`, `pi05_exp.py`, `oft_exp.py`, `navila_exp.py`, `memvla_exp.py` 등

모두 `InferenceConfig`를 상속하며, `_get_response()` 메서드를 오버라이드하여 모델별 추론 로직을 구현한다. `playground/benchmarks/` 디렉토리에는 벤치마크×모델 조합별 런치 스크립트가 있다 (예: `libero_cogact.py`, `libero_pi0.py`).

이 구조는 우리의 `ModelServer` → `PredictModelServer` 패턴과 유사하지만, dexbotic에서는 모델 서버가 항상 Flask 웹서버로 실행된다는 점이 다르다.

## 코드 품질/성숙도

| 측면 | 평가 |
|------|------|
| 타입 힌트 | 부분적. BaseEvaluator, BaseVLAAgent에는 있으나 하위 클래스에서 불일치 |
| 테스트 | 없음. 테스트 디렉토리 자체가 없음 |
| CI/CD | 없음 |
| 문서화 | README 수준. docstring은 일부 존재 |
| 코드 중복 | 높음. 특히 evaluator 간 에피소드 루프 |
| 오타/미비 | 간헐적 (예: `pormpt` → `prompt`) |
| 의존성 관리 | conda 환경 + pip. requirements 파일 분산 |
| 전반적 성숙도 | **연구 프로토타입 수준**. 빠르게 동작하지만 프로덕션 품질은 아님 |

## 시사점

### 채택할 패턴

1. **액션 청킹 관리 패턴**: dexbotic의 `replan_step` 기반 버퍼 관리 패턴은 깔끔하다. 다만 dexbotic는 이를 클라이언트 사이드(Agent 계층)에서 처리하는 반면, 우리 설계에서는 **서버 사이드**(`PredictModelServer.chunk_size`)로 이동하여 관심사를 더 깔끔하게 분리한다. 벤치마크(클라이언트)는 매 step마다 단일 action만 받으며, 청킹 로직을 알 필요가 없다.

2. **Adaptive Ensemble의 코사인 유사도 기반 가중치** → 우리 `PredictModelServer.action_ensemble=callable` 옵션에 매핑. 코사인 유사도 기반 앙상블 로직을 callable 함수로 구현하여 전달하면 된다. `"average"`(고정 가중치 temporal ensemble) 외의 전략 옵션으로 제공 가능.

3. **벤치마크별 관측 전처리 분리**: 에이전트 서브클래스가 관측 전처리만 담당하는 패턴. 우리의 `Benchmark.make_obs()` → 표준 dict 포맷 → 모델 서버 입력 변환 파이프라인의 참고가 된다.

4. **YAML 설정의 단순함**: 지나치게 복잡한 Hydra config group 대신 flat YAML이 사용자 친화적 → 우리의 YAML 설정 체계와 일치하는 방향.

### 회피할 패턴

1. **에피소드 루프의 서브클래스 하드코딩**: 가장 큰 설계 문제. Evaluator마다 거의 동일한 루프를 재구현한다 → 우리의 **EpisodeRunner**(SyncEpisodeRunner/AsyncEpisodeRunner)가 "어떻게"를 분리하여 해결. Benchmark는 "무엇"만 정의.

2. **모놀리식 Docker 이미지**: 단일 이미지에 모든 벤치마크 환경을 패키징하면 이미지 크기가 폭발하고, 하나의 벤치마크 의존성 변경이 전체 빌드를 트리거한다 → 우리의 **벤치마크별 독립 Docker 이미지** 전략.

3. **동기 HTTP REST 통신**: 프레임마다 HTTP 연결 + PNG 인코딩은 성능 병목 → 우리의 **WebSocket + msgpack Protocol**이 persistent connection + 바이너리 직렬화로 개선.

4. **통합 CLI 부재**: 벤치마크별로 다른 진입점은 사용자 경험을 해친다 → 우리의 **Orchestrator** CLI가 통합 진입점 제공.

5. **서버 threaded=False**: 동시 요청 처리 불가. 우리 설계에서는 비동기 서버가 기본이어야 한다.

### 우리 설계 컴포넌트와의 매핑 요약

| dexbotic 요소 | 우리 설계 컴포넌트 | 비고 |
|--------------|-------------------|------|
| `BaseEvaluator` ABC | **Benchmark ABC** | 우리는 "무엇(Benchmark)"과 "어떻게(EpisodeRunner)"를 분리 |
| `BaseVLAAgent` (HTTP 클라이언트 + 청킹) | **Connection** (클라이언트 라이브러리) | 프레임워크 제공, 사용자 구현 불필요 |
| YAML config | **YAML 설정 체계** | 유사한 접근 |
| Evaluator 서브클래스 내 에피소드 루프 | **EpisodeRunner** (Sync/Async) | 벤치마크별 루프 중복 제거 |
| 모놀리식 Docker | **벤치마크별 독립 Docker 이미지** | 격리 및 확장성 개선 |
| HTTP REST (Flask) | **WebSocket + msgpack Protocol** | persistent connection + 바이너리 직렬화 |
| 벤치마크별 별도 진입점 | **Orchestrator** CLI | 통합 진입점 |
| 벤치마크별 JSON 결과 | **Result Collector** | 에피소드/태스크/벤치마크 통합 메트릭 |
| `replan_step` 기반 클라이언트 청킹 | `PredictModelServer.chunk_size` (서버 사이드) | 서버에서 관리 |
| Adaptive Ensemble | `PredictModelServer.action_ensemble=callable` | callable로 전달 |

### 열린 질문

1. **관측 전처리는 어디서 해야 하는가?** dexbotic는 클라이언트 측(에이전트 서브클래스)에서 한다. 우리 설계에서는 `Benchmark.make_obs()`가 환경의 raw observation을 model server에 전송할 dict로 변환하는 역할을 담당한다. 모델 서버 측의 추가 전처리(이미지 리사이즈 등)는 `predict()` 내부에서 처리하는 것이 적합한가?

2. **결과 포맷 표준화**: dexbotic는 벤치마크별로 다른 JSON 구조를 사용한다. 우리의 **Result Collector**가 통합 결과 스키마를 정의할 때, 벤치마크별 커스텀 메트릭과 공통 메트릭(성공률, 에피소드 길이 등)을 어떻게 공존시킬 것인가?

3. ~~모델 서버 다형성의 범위: 우리의 `ModelServer` ABC가 충분히 유연한가?~~ → **해결됨**: `PredictModelServer`(blocking 추론), `BatchPredictModelServer`(배치 추론), `ModelServer`(완전 async) 3단계 추상화로 대부분의 모델을 수용. dexbotic의 모델별 서버 코드(`cogact_exp.py`, `pi0_exp.py` 등)는 각각 `PredictModelServer`의 서브클래스로 구현하면 된다.
