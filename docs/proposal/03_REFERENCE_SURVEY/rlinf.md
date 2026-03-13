# RLinf — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/RLinf/RLinf |
| Latest Commit | `c7f1607` / 2026-02-14 |
| 규모 | Python 399개 파일, ~94,323 라인 |
| 라이선스 | Apache License 2.0 |
| ArXiv | 2509.15965 |
| 주요 목적 | 분산 강화학습 인프라 — VLA 모델(Pi0, OpenVLA, GR00T 등)을 PPO/GRPO/SAC 알고리즘으로 RL 학습. Ray 기반 스케줄러 + NCCL 집합통신 |
| 우리 프로젝트에서의 참고 포인트 | 01_PROPOSAL.md: "환경 wrapper 설계 참고" |

## 프로젝트 구조

```
RLinf/
├── rlinf/
│   ├── scheduler/                      # Ray 기반 분산 스케줄러
│   │   ├── worker/
│   │   │   ├── worker.py               # Worker(metaclass=WorkerMeta) — 1,263줄, 핵심 기반 클래스
│   │   │   └── worker_group.py         # WorkerGroup — Worker 그룹 관리
│   │   ├── cluster.py                  # Cluster — 노드/GPU 자원 관리
│   │   ├── channel/                    # Channel — 워커간 데이터 전달 큐
│   │   ├── collective/                 # NCCL 집합통신 (send/recv/broadcast)
│   │   └── manager/                    # WorkerManager — 워커 등록/주소 관리
│   ├── runners/                        # 학습 루프 오케스트레이션
│   │   ├── embodied_runner.py          # EmbodiedRunner — 핵심 학습 루프 (292줄)
│   │   ├── async_embodied_runner.py    # AsyncEmbodiedRunner — 비동기 변형
│   │   ├── reasoning_runner.py         # ReasoningRunner — 추론 RL
│   │   ├── agent_runner.py             # AgentRunner — 도구 사용 에이전트
│   │   └── sft_runner.py              # SFTRunner — 지도학습
│   ├── algorithms/                     # 알고리즘 레지스트리
│   │   ├── ppo.py                      # PPO (GAE, 클리핑, actor-critic)
│   │   ├── grpo.py                     # GRPO (Group Relative Policy Optimization)
│   │   ├── sac.py                      # SAC (Soft Actor-Critic)
│   │   └── crossq.py                  # CrossQ (저차원 제어)
│   ├── models/                         # 모델 레지스트리
│   │   ├── openpi/                     # Pi0/Pi0.5 — Flow-matching VLA
│   │   ├── openvla/                    # OpenVLA / OpenVLA-OFT — Autoregressive VLA
│   │   ├── gr00t/                      # NVIDIA GR00T
│   │   ├── dexbotic_pi0/              # Dexbotic Pi0 (dexterous manipulation)
│   │   └── mlp/                        # MLP 베이스라인
│   ├── envs/                           # 환경 레지스트리
│   │   ├── libero/                     # LIBERO (spatial, object, goal, 10, 90, 130)
│   │   ├── maniskill/                  # ManiSkill (GPU 병렬화)
│   │   ├── calvin/                     # CALVIN (언어 조건)
│   │   ├── isaaclab/                   # NVIDIA IsaacLab
│   │   ├── behavior/                   # BehaviorVR
│   │   ├── gs_env/                     # Gaussian Splatting 환경
│   │   ├── franka/                     # FrankaSim
│   │   ├── real/                       # 실제 로봇 환경
│   │   └── venv/                       # Vectorized env (DummyVectorEnv, SubprocVectorEnv)
│   ├── workers/                        # 워커 구현체
│   │   ├── actor/                      # FSDP/Megatron actor, SAC policy
│   │   ├── rollout/                    # HF/vLLM/SGLang 추론 백엔드
│   │   ├── env/                        # 환경 워커 (Ray 기반)
│   │   ├── inference/                  # FSDP/Megatron 추론 워커
│   │   └── reward/                     # 보상 모델 워커
│   ├── data/
│   │   └── embodied_io_struct.py       # EnvOutput, ChunkStepResult, Trajectory (522줄)
│   └── utils/                          # 로깅, 메트릭, 분산 유틸리티
└── examples/
    └── embodiment/config/              # Hydra YAML 예시 설정
```

## 핵심 분석

### 벤치마크 추상화

**환경 레지스트리 패턴 (`@EnvRegistry.register()`):**

```python
# rlinf/envs/ — 각 환경은 데코레이터로 등록
@EnvRegistry.register("libero_spatial")
class LiberoSpatialEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, ...): ...
    def reset(self, **kwargs): ...
    def step(self, action): ...
```

지원 환경: LIBERO (6 변형), ManiSkill, CALVIN, IsaacLab, BehaviorVR, GS-Env, FrankaSim, 실제 로봇.

**공통 인터페이스**: 모든 환경은 `gym.Env` 또는 자체 래퍼를 상속하되, 최종적으로 `EnvOutput` dataclass로 통일된 출력:

```python
@dataclass(kw_only=True)
class EnvOutput:
    obs: dict[str, Any]          # main_images, wrist_images, extra_view_images, states, task_descriptions
    final_obs: Optional[dict]
    dones: Optional[torch.Tensor]         # [B]
    terminations: Optional[torch.Tensor]  # [B]
    truncations: Optional[torch.Tensor]   # [B]
    rewards: Optional[torch.Tensor]       # [B]
    intervene_actions: Optional[torch.Tensor]  # [B] — 인간 개입 액션
    intervene_flags: Optional[torch.Tensor]    # [B] — 개입 플래그
```

**Vectorized Environment** — Tianshou 유래의 `BaseVectorEnv` / `DummyVectorEnv` / `SubprocVectorEnv` 패턴:
- `DummyVectorEnv`: for-loop 기반 순차 실행
- `SubprocVectorEnv`: multiprocessing.Pipe 기반 병렬 실행
- `EnvWorker` ABC → `DummyEnvWorker`, `SubprocEnvWorker` 구현
- ManiSkill은 GPU 병렬화된 자체 벡터화 사용 (`ManiskillEnv`가 내부적으로 `gym.make(num_envs=N)`)

→ **우리 프로젝트와의 관계**: 환경마다 gym.Env을 직접 상속하는 패턴. Docker 격리 없이 동일 프로세스 내에서 실행. 우리는 Docker 기반 격리가 필요하므로 이 패턴을 직접 채택하기보다는 참고용.

### 모델 통합 패턴

**모델 레지스트리 (`@ModelRegistry.register()`):**

```python
@ModelRegistry.register("openpi")
class OpenPIModel:
    def __init__(self, cfg): ...
    def forward(self, batch): ...           # 학습 시 loss
    def predict_action(self, obs): ...      # 추론 시 action
```

지원 모델: OpenPI (Pi0/Pi0.5), OpenVLA / OpenVLA-OFT, GR00T, Dexbotic Pi0, MLP.

**핵심 차이점 (starVLA 대비):**
- starVLA는 `predict_action → {"normalized_actions": ndarray}` 형식
- RLinf는 RL 학습을 위해 `logprobs`, `values`, `forward_inputs`를 함께 반환 (`ChunkStepResult`)
- starVLA는 WebSocket 서버 기반 추론, RLinf는 Ray actor 기반 in-cluster 추론

### 에피소드 실행 루프

**EmbodiedRunner** — 핵심 학습 루프 (292줄):

```python
class EmbodiedRunner:
    def __init__(self, cfg, actor, rollout, env, critic=None, reward=None):
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")

    def run(self):
        for step in range(start_step, max_steps):
            # 1. 가중치 동기화: actor → rollout
            self.update_rollout_weights()

            # 2. 롤아웃 생성: env ↔ rollout 상호작용
            env_handle = self.env.interact(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel)
            rollout_handle = self.rollout.generate(
                input_channel=self.env_channel,
                output_channel=self.rollout_channel,
                actor_channel=self.actor_channel)
            self.actor.recv_rollout_trajectories(
                input_channel=self.actor_channel).wait()

            # 3. Advantage/Return 계산
            self.actor.compute_advantages_and_returns().wait()

            # 4. Actor 학습
            self.actor.run_training().wait()
```

**Channel 기반 데이터 흐름:**
- `env_channel`: 환경 → 롤아웃 (관측 전달)
- `rollout_channel`: 롤아웃 → 환경 (액션 전달)
- `actor_channel`: 롤아웃 → 액터 (궤적 전달)

→ **우리 프로젝트와의 관계**: 우리는 학습이 아닌 평가 프레임워크이므로 이 actor-critic 루프가 직접 필요하진 않지만, Channel 기반 env↔model 상호작용 패턴은 참고할 만함.

### 통신 구조

**Ray 기반 분산 통신** — starVLA의 WebSocket과 근본적으로 다른 접근:

| 측면 | RLinf | starVLA |
|------|-------|---------|
| 통신 | Ray actor + NCCL | WebSocket + msgpack |
| 스케일 | 멀티노드 GPU 클러스터 | 단일 서버 + 클라이언트 |
| 목적 | RL 학습 (고빈도 tensor 전송) | 추론 서빙 (요청-응답) |
| 직렬화 | NCCL (GPU 텐서), GLOO (CPU 객체) | msgpack + numpy |

**Worker 통신 API:**
```python
class Worker(metaclass=WorkerMeta):
    def send(self, object, dst_group_name, dst_rank, async_op=False): ...
    def recv(self, src_group_name, src_rank, async_op=False): ...
    def send_tensor(self, tensor, dst_group_name, dst_rank): ...   # NCCL 최적화
    def recv_tensor(self, tensor, src_group_name, src_rank): ...   # in-place
    def broadcast(self, object, groups, src=None): ...
```

- GPU 텐서: NCCL 직접 전송 (zero-copy)
- CPU 객체: GLOO를 통해 직렬화 전송
- 비동기 send/recv 지원 (`async_op=True`)

### 환경 격리

**Docker 격리 없음.** 환경은 Ray actor 프로세스 내에서 직접 실행.

- `EnvWorker` (Ray actor): 환경 인스턴스를 보유하고 `interact()` / `evaluate()` 메서드 노출
- 환경 벡터화: `DummyVectorEnv` 또는 `SubprocVectorEnv`로 단일 워커 내 다중 환경 운영
- ManiSkill: GPU 병렬화로 단일 GPU에서 수백 개 환경 동시 실행

**환경 워커 패턴:**
```python
# rlinf/workers/env/env_worker.py
class EnvWorker(Worker):       # Ray actor로 동작
    def init_worker(self):
        self.train_env = create_env(self.cfg.env.train)  # 벡터화된 환경
        self.eval_env = create_env(self.cfg.env.eval)

    def interact(self, input_channel, output_channel):
        # input_channel에서 액션 수신 → env.step → output_channel로 관측 전송
        ...
```

→ **시사점**: 우리 프로젝트는 Docker 격리가 필수인데, RLinf는 성능 최적화 위주로 in-process 실행. 벡터화 환경 패턴 자체는 참고하되, 격리 전략은 다르게 가야 함.

### 설정/CLI

**Hydra + OmegaConf 기반 계층적 설정:**

```yaml
# examples/embodiment/config/libero_spatial_ppo_openpi_quickstart.yaml
defaults:
  - env/libero_spatial@env.train       # 환경 설정 합성
  - env/libero_spatial@env.eval
  - model/pi0@actor.model              # 모델 설정 합성
  - training_backend/fsdp@actor.fsdp_config

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all             # GPU 배치 전략

algorithm:
  clip_ratio_high: 0.2                # PPO 하이퍼파라미터
  gamma: 0.99
  gae_lambda: 0.95
  reward_type: chunk_level

env:
  train:
    total_num_envs: 32
    max_episode_steps: 240
  eval:
    total_num_envs: 500
```

**특징:**
- `defaults` 리스트로 설정 합성 (`@` 문법으로 중첩 키 매핑)
- `${oc.env:VAR}`, `${actor.model.precision}` 등 변수 참조
- `component_placement`로 워커-GPU 매핑 선언
- 별도 CLI 도구 없이 `python -m rlinf.train config=...` 형태

### 결과 수집

**MetricLogger + TensorBoard/WandB/SwanLab:**

```python
self.metric_logger = MetricLogger(cfg)
# 학습 중:
self.metric_logger.log(env_metrics, step)      # env/success_rate 등
self.metric_logger.log(training_metrics, step)  # train/loss 등
self.metric_logger.log(time_metrics, step)      # time/step 등
```

- `compute_evaluate_metrics()`: 평가 에피소드 결과를 집계
- `print_metrics_table()`: 비동기 로그 큐 기반 콘솔 출력
- 비디오 레코딩: 환경 설정에서 `video_cfg.save_video: True`

→ **우리 프로젝트와의 관계**: 학습 메트릭 위주. 우리는 벤치마크 평가 결과를 구조화된 JSON/리더보드 형식으로 수집해야 하므로 직접 적용은 어려움.

## 프로젝트별 심층 분석

### 1. 레지스트리 패턴 (Algorithm / Model / Env)

RLinf는 **3중 레지스트리** 패턴을 일관되게 사용:

```python
# 패턴 공통:
class Registry:
    _registry = {}
    @classmethod
    def register(cls, name):
        def decorator(klass):
            cls._registry[name] = klass
            return klass
        return decorator
    @classmethod
    def get(cls, name):
        return cls._registry[name]

# 사용:
AlgorithmRegistry.register("ppo")(PPOAlgorithm)
ModelRegistry.register("openpi")(OpenPIModel)
EnvRegistry.register("libero_spatial")(LiberoSpatialEnv)
```

**확장 모듈 로딩** (`EXT_MODULE` 환경변수):
```python
# Worker.__init__ 시 자동 호출
def _load_user_extensions(self):
    ext_module = importlib.import_module(os.environ["EXT_MODULE"])
    ext_module.register()  # 사용자 정의 환경/모델/알고리즘 등록
```

→ lm-eval-harness의 `--include_path`와 유사한 확장 메커니즘. 코어 수정 없이 커스텀 컴포넌트 추가 가능.

### 2. Trajectory 데이터 구조 (RL 특화)

```python
@dataclass
class Trajectory:
    max_episode_length: int
    model_weights_id: str
    actions: torch.Tensor              # [T, B, action_dim]
    rewards: torch.Tensor              # [T, B, 1]
    terminations: torch.Tensor         # [T, B, 1]
    truncations: torch.Tensor
    dones: torch.Tensor
    prev_logprobs: torch.Tensor        # [T, B, action_dim] — RL 학습용
    prev_values: torch.Tensor          # [T, B, 1] — critic 예측
    forward_inputs: dict               # 모델 forward에 필요한 추가 입력
    intervene_flags: torch.Tensor      # 인간 개입 플래그
    curr_obs: dict                     # 현재 관측
    next_obs: dict                     # 다음 관측

    def extract_intervene_traj(self):  # 인간 개입 구간만 추출
        ...
```

**EmbodiedRolloutResult** → `to_trajectory()` 변환:
- 청크 스텝 결과를 수집하다가 에피소드 완료 시 `Trajectory`로 스택
- `to_splited_trajectories(split_size)`: 다중 actor 워커로 분배

→ **시사점**: RL 학습 데이터 구조이므로 우리 평가 프레임워크에는 불필요. 하지만 관측(`obs`)의 구조화 방식은 참고:
  - `main_images: [N_ENV, H, W, C]`
  - `wrist_images: [N_ENV, H, W, C]` 또는 `[N_ENV, N_IMG, H, W, C]`
  - `states: Tensor`
  - `task_descriptions: list[str]`

### 3. Worker 메타클래스와 에러 처리

```python
class WorkerMeta(type):
    """Metaclass to capture failures in worker classes."""
    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = cls._catch_failure_for_cls_func(name, attr_name, attr_value)
        return super().__new__(cls, name, bases, attrs)
```

- `SystemExit`을 `RuntimeError`로 변환하여 Ray actor 사망 방지
- 동기/비동기 함수 모두 래핑
- 프라이빗 메서드(`_`로 시작)는 제외

→ **시사점**: 분산 환경에서의 견고한 에러 처리 패턴. 우리 Docker 워커에서도 유사한 방어적 래핑 필요.

### 4. ManiSkill GPU 병렬화 환경

```python
class ManiskillEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, ...):
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args)  # num_envs가 init_params에 포함
        # GPU 텐서 기반 관측/액션 — CPU 전송 오버헤드 최소화
```

- `gym.make(id="PickCube-v1", num_envs=500)`: 단일 GPU에서 수백 개 환경 동시 실행
- 관측/액션이 GPU 텐서로 직접 반환 → NCCL로 모델 워커에 전송
- `extract_termination_from_info()`: success/fail 정보에서 termination 추출

### 5. 비동기 학습 패턴 (AsyncEmbodiedRunner)

실제 로봇이나 고지연 환경을 위한 비동기 변형:
- 롤아웃과 학습이 동시에 진행
- 가중치 동기화 간격 (`weight_sync_interval`) 조절로 staleness 관리
- `AsyncEnvWorker`: 환경 상호작용이 블로킹되지 않음

→ **시사점**: 우리 프레임워크의 실제 로봇 평가 시나리오에서 유사한 비동기 패턴 필요할 수 있음.

## 코드 품질/성숙도

| 측면 | 평가 |
|------|------|
| 타입 힌트 | 양호 — `dict[str, Any]`, `Optional[torch.Tensor]` 등 현대적 문법. `dataclass(kw_only=True)` 활용 |
| 테스트 | 존재 확인 필요 (주요 기능에 대한 테스트는 미확인) |
| CI | GitHub Actions 추정 |
| 문서화 | ReadTheDocs 문서 (영문/중문), ArXiv 논문, DeepWiki 연동 |
| 코드 품질 | 높음 — 일관된 레지스트리 패턴, 명확한 관심사 분리, 포괄적 docstring (Worker 클래스 등) |
| 성숙도 | 중후기 단계. 활발한 커뮤니티, PyPI 배포, 다수 모델/환경 통합 완료 |

## 시사점

### 채택할 패턴

1. **3중 레지스트리 패턴** — Algorithm/Model/Env를 동일한 `@Registry.register()` 데코레이터로 등록. 우리도 Benchmark/ModelServer 레지스트리에 동일 패턴 적용.
2. **`EXT_MODULE` 확장 메커니즘** — 코어 수정 없이 외부 모듈의 `register()` 함수를 호출하여 커스텀 컴포넌트 등록. 우리 Benchmark 플러그인 시스템의 참고 모델. lm-eval-harness의 `--include_path`와 결합하면 강력한 확장성.
3. **EnvOutput dataclass의 관측 필드 관례** — `main_images`, `wrist_images`, `states`, `task_descriptions`는 VLA 생태계의 사실상 표준 필드셋. 다만 우리 설계는 인터페이스 수준에서 고정 스키마를 강제하지 않고 `dict[str, Any]`를 채택한다. 각 Benchmark의 `make_obs()`가 raw observation을 모델 서버가 기대하는 dict 구조로 변환하며, RLinf의 필드 네이밍은 문서/예시에서 참조하는 **관례**(convention)로 활용한다. 이 "범용 슈퍼클래스 + 벤치마크별 어댑터" 접근이 고정 스키마보다 유연하다.
4. **EnvOutput → StepResult 매핑** — RLinf의 `EnvOutput`(obs, dones, rewards 등)은 우리 `StepResult`(obs, reward, done, info)에 대응. 배치 차원과 RL 전용 필드(`intervene_actions` 등)를 제거하면 동일한 개념.
5. **Vectorized Env → Docker 격리 + Orchestrator** — RLinf의 `DummyVectorEnv`/`SubprocVectorEnv`는 다중 환경 병렬 실행을 담당. 우리는 Docker 컨테이너 격리 + Orchestrator가 이 역할을 대체하며, 환경 간 의존성 충돌을 근본적으로 해결.
6. **분산 워커 관리 → Orchestrator** — RLinf의 `WorkerManager`(워커 등록/주소 관리)와 `Cluster`(자원 관리)는 RL 학습용 분산 인프라. 우리 Orchestrator는 이를 평가 시나리오에 맞게 단순화 — Docker 기동, 서버 연결, runner 선택, 에피소드 반복.
7. **Hydra 설정 합성** — `defaults` 리스트로 환경/모델/알고리즘 설정을 독립 파일로 분리 후 합성. 복잡한 설정의 관리성 향상.
8. **WorkerMeta 에러 래핑** — 분산 워커에서 `SystemExit` → `RuntimeError` 변환. Docker 컨테이너 내 평가 프로세스의 안정성 향상에 응용 가능.
9. **비동기 로깅 큐** — `queue.Queue` + 백그라운드 스레드로 메트릭 출력을 비블로킹 처리.

### 회피할 패턴

1. **Ray 의존성** — 평가 프레임워크에 Ray 수준의 분산 인프라는 과도. 우리는 Docker + REST/WebSocket이 적합.
2. **NCCL 직접 통신** — 학습 시 GPU 텐서 전송에 최적이지만, 우리의 모델 서버 ↔ 벤치마크 통신에는 msgpack/REST가 적합.
3. **in-process 환경 실행** — Docker 격리 없이 환경을 Ray actor 내에서 직접 실행. 재현성과 의존성 충돌 위험.
4. **RL 학습 전용 데이터 구조** — `prev_logprobs`, `prev_values` 등 RL 학습에만 필요한 필드. 우리 평가 프레임워크에서는 `obs → action → reward/done` 수준의 단순한 구조가 적합.

### 열린 질문

1. ~~RLinf의 `EnvOutput.obs` 구조를 우리 `Observation` 타입의 기준으로 삼을 것인가?~~ → **해결됨**: 우리 설계는 `dict[str, Any]`(고정 스키마 없음)를 채택했으며, `Benchmark.make_obs()`가 벤치마크별 변환을 담당한다. RLinf의 필드 네이밍(`main_images`, `wrist_images`, `states`, `task_descriptions`)은 문서와 예시에서 관례적으로 참조한다.
2. Hydra 설정 합성 패턴을 우리 프레임워크에 도입할 것인가? 아니면 lm-eval-harness 스타일의 단순 YAML + CLI 오버라이드를 유지할 것인가?
3. `EXT_MODULE` 방식의 플러그인 로딩 vs lm-eval-harness의 `--include_path` 방식 — 우리 프레임워크에 어느 것이 적합한가? 둘 다 지원할 수 있는가?
4. Vectorized Environment 패턴 (`DummyVectorEnv` / `SubprocVectorEnv`)을 Docker 격리와 어떻게 결합할 것인가? Docker 컨테이너 내에서 SubprocVectorEnv를 사용하고, 컨테이너 자체가 격리 레이어가 되는 구조?

