# BEHAVIOR-1K (OmniGibson) — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/StanfordVL/OmniGibson |
| Last Commit | 2026-02-13 |
| 규모 | 561 Python 파일, ~154,522 LOC |
| 주요 목적 | BEHAVIOR-1K 벤치마크를 위한 가정환경 시뮬레이션 플랫폼. NVIDIA Isaac Sim 기반 물리 시뮬레이션, 로봇 제어, 태스크 정의, 평가 인프라 통합 제공 |
| 우리 프로젝트에서의 참고 포인트 | BEHAVIOR-1K를 벤치마크로 지원할 때의 환경 인터페이스 및 평가 프로토콜 참고 |

## 프로젝트 구조

```
BEHAVIOR-1K/
├── OmniGibson/                         # 핵심 시뮬레이션 패키지
│   └── omnigibson/
│       ├── envs/
│       │   ├── env_base.py             # Environment(gym.Env) — Gymnasium 호환 환경 (905줄)
│       │   ├── env_wrapper.py          # EnvironmentWrapper — 환경 래퍼 ABC + 레지스트리
│       │   ├── data_wrapper.py         # 데이터 수집 래퍼
│       │   └── metrics_wrapper.py      # 메트릭 래퍼
│       ├── learning/
│       │   ├── eval.py                 # Evaluator 클래스 — 평가 오케스트레이션 (497줄)
│       │   ├── eval_with_jobqueue.py   # 분산 평가 변형 (274줄)
│       │   ├── policies.py             # LocalPolicy + WebsocketPolicy (77줄)
│       │   ├── utils/
│       │   │   ├── network_utils.py    # WebsocketClientPolicy + WebsocketPolicyServer + msgpack numpy (284줄)
│       │   │   ├── eval_utils.py       # 로봇별 카메라/관절 상수, 태스크 매핑, obs 전처리 (314줄)
│       │   │   ├── score_utils.py      # 최종 Q-score 계산, 제출 결과 집계 (204줄)
│       │   │   └── config_utils.py     # OmegaConf resolver 등록
│       │   ├── wrappers/               # 환경 래퍼 구현체들
│       │   │   ├── default_wrapper.py  # DefaultWrapper — 카메라 해상도/modality 설정
│       │   │   ├── rgb_low_res_wrapper.py
│       │   │   └── challenge_submissions/  # 챌린지 참가팀별 커스텀 래퍼
│       │   └── configs/                # Hydra YAML 설정
│       │       ├── base_config.yaml    # 기본 설정 (17줄)
│       │       ├── policy/             # local.yaml, websocket.yaml
│       │       ├── robot/              # r1pro.yaml, a1.yaml
│       │       └── task/               # behavior.yaml
│       ├── metrics/
│       │   ├── metric_base.py          # MetricBase ABC — start/step/end 콜백 (19줄)
│       │   ├── agent_metric.py         # AgentMetric — 로봇 이동 거리 추적 (61줄)
│       │   └── task_metric.py          # TaskMetric — q_score + 시간 메트릭 (55줄)
│       ├── tasks/
│       │   └── task_base.py            # BaseTask(GymObservable, Registerable) ABC (458줄)
│       ├── robots/                     # 로봇 정의 (R1Pro, Fetch, Tiago 등)
│       ├── controllers/                # 로봇 컨트롤러 (IK, OSC, Joint 등)
│       └── utils/                      # 유틸리티 라이브러리
├── bddl3/                              # 태스크 정의 언어 (BDDL)
├── joylo/                              # 텔레오퍼레이션 하드웨어 인터페이스
├── eval-jobqueue/                      # 분산 평가 작업 큐
│   ├── jobqueue.py                     # FastAPI 작업 큐 서버 (468줄)
│   ├── generate_jobs.py                # 작업 생성 스크립트 (331줄)
│   └── check_results.py               # 결과 검증
└── docker/
    ├── Dockerfile                      # 메인 Docker 이미지 (Isaac Sim + CUDA + conda)
    └── submission.Dockerfile           # 챌린지 제출용 Docker — 모델 서버 컨테이너
```

## 핵심 분석

### 벤치마크 추상화

**단일 벤치마크 플랫폼**이지, 다중 벤치마크 프레임워크가 아님. Environment(gym.Env)가 유일한 환경 인터페이스:

```python
class Environment(gym.Env, GymObservable, Recreatable):
    def step(self, action, n_render_iterations=1) -> (obs, reward, terminated, truncated, info)
    def reset(self, get_obs=True) -> (obs, info)
    def load(self)  # scene → objects → robots → task → sensors 순서
```

태스크는 `REGISTERED_TASKS` 딕셔너리 레지스트리를 통해 관리. `BaseTask(GymObservable, Registerable, ABCMeta)`가 ABC:
- `_create_termination_conditions()` / `_create_reward_functions()` — 서브클래스에서 구현
- `BehaviorTask` — BDDL 기반 태스크 정의의 실제 구현체
- `verify_scene_and_task_config()` — 태스크-씬 호환성 검증

환경 래퍼 시스템: `EnvironmentWrapper(Wrapper, Registerable)` — `REGISTERED_ENV_WRAPPERS` 레지스트리로 관리. 래퍼를 통한 관찰 공간 수정 (카메라 해상도, modality 추가 등). 챌린지 참가팀은 자신만의 래퍼를 정의하여 관찰 전처리를 커스터마이징.

### 모델 통합 패턴

두 가지 정책 인터페이스:

1. **LocalPolicy** — 프로세스 내 직접 실행. `forward(obs) → action`, 정책 미설정 시 zero-action 반환
2. **WebsocketPolicy** — WebSocket 원격 정책:
   - `forward(obs)` → obs를 numpy로 변환 → WebSocket 전송 → action 수신 → torch 반환
   - `need_new_action` 플래그로 액션 캐싱 지원 (새 액션 불필요 시 이전 액션 재사용)
   - `update_host(host, port)` — 런타임 중 서버 주소 변경 가능

정책 로딩은 Hydra `instantiate(self.cfg.model)` — DI 패턴:
```yaml
# configs/policy/websocket.yaml
model:
  _target_: omnigibson.learning.policies.WebsocketPolicy
  host: 0.0.0.0
  port: 8000
```

### 에피소드 실행 루프

`Evaluator` 클래스 (497줄)가 전체 평가를 오케스트레이션:

```python
class Evaluator:
    def __init__(self, cfg):
        self.env = self.load_env(env_wrapper=cfg.env_wrapper)
        self.policy = self.load_policy()       # Hydra instantiate
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()     # [AgentMetric, TaskMetric]
    
    def step(self) -> (terminated, truncated):
        action = self.policy.forward(obs=self.obs)
        obs, _, terminated, truncated, info = self.env.step(action)
        self.obs = self._preprocess_obs(obs)
        for metric in self.metrics: metric.step_callback(self.env)
        return terminated, truncated
    
    def reset(self):
        self.obs = self._preprocess_obs(self.env.reset()[0])
        for metric in self.metrics: metric.start_callback(self.env)
        self.policy.reset()
```

**메인 루프** (`if __name__ == "__main__"`):
1. Hydra config 로드 (base_config.yaml + CLI overrides)
2. 테스트 인스턴스 목록 결정 (train/test/hidden)
3. `with Evaluator(config) as evaluator:` — 컨텍스트 매니저 (SIGINT 처리)
4. 인스턴스별 루프: `reset() → load_task_instance(idx) → reset() → while not done: step()`
5. 에피소드 종료 시: `end_callback → gather_results → JSON 저장`

**관찰 전처리** (`_preprocess_obs`):
- obs dict 평탄화 (nested → `::` 구분 키)
- 카메라 상대 pose 계산 (base 기준)
- 태스크 ID 추가

### 통신 구조

**WebSocket + msgpack** 기반 (openpi에서 adapted):

```python
# network_utils.py — 서버 + 클라이언트 + msgpack numpy 직렬화 통합
class WebsocketClientPolicy:
    def act(self, obs: Dict) -> th.Tensor:
        data = self._packer.pack(obs)         # msgpack + numpy 직렬화
        self._ws.send(data)
        response = self._ws.recv()
        action_dict = unpackb(response)       # {"action": np.ndarray, "server_timing": {...}}
        return th.from_numpy(action_dict["action"])
    
    def reset(self):
        self._ws.send(self._packer.pack({"reset": True}))

class WebsocketPolicyServer:
    async def _handler(self, websocket):
        await websocket.send(packer.pack(self._metadata))  # 연결 시 메타데이터 전송
        while True:
            result = unpackb(await websocket.recv())
            if "reset" in result: self._policy.reset(); continue
            action = self._policy.act(obs)
            await websocket.send(packer.pack({"action": action.cpu().numpy(), "server_timing": {...}}))
```

**핵심 설계 포인트**:
- **Health check**: WebSocket 연결 전 HTTP `/healthz` 엔드포인트로 서버 준비 상태 확인
- **API key 인증**: `Authorization: Api-Key {key}` 헤더 지원
- **msgpack numpy 직렬화**: `pack_array`/`unpack_array` — `{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": tuple}` 형식. `functools.partial(msgpack.Packer, default=pack_array)`로 커스텀 Packer 생성
- **서버 타이밍**: 응답에 `server_timing: {infer_ms, prev_total_ms}` 포함 — 성능 프로파일링 지원
- **에러 핸들링**: 디버그 모드에서 traceback 전송, `INTERNAL_ERROR` 코드로 연결 종료

### 환경 격리

**Docker 기반 이중 구조**:

1. **메인 Docker** (`docker/Dockerfile`, 110줄): NVIDIA base → Isaac Sim 의존성 → Miniconda → BEHAVIOR conda env → CUDA toolkit 설치/제거 (빌드 시만 필요). `VOLUME ["/data", "/cache"]`로 에셋/캐시 마운트. SLURM 배치 제출 지원 (`sbatch_example.sh`).

2. **Submission Docker** (`docker/submission.Dockerfile`, 45줄): 챌린지 참가팀용 **최소 컨테이너**. `bddl + OmniGibson[eval]`만 설치. 기본 CMD:
```python
WebsocketPolicyServer(LocalPolicy(action_dim=23)).serve_forever()
```
참가팀은 이 기본 CMD를 자신의 모델 서버로 교체. **핵심 설계: 평가 클라이언트(시뮬레이터)와 모델 서버(참가팀)를 별도 컨테이너로 분리.**

### 설정/CLI

**Hydra + OmegaConf** 기반 설정 시스템:

```yaml
# base_config.yaml
defaults:
  - _self_
  - robot: r1pro        # configs/robot/r1pro.yaml
  - task: behavior       # configs/task/behavior.yaml
  - policy: ???          # 반드시 CLI에서 지정 (local 또는 websocket)

env_wrapper:
  _target_: omnigibson.learning.wrappers.RGBLowResWrapper
headless: true
max_steps: null          # null → human demo 평균 완료 스텝 × 2
log_path: ???
```

- `policy: ???` — **필수 인자**로 강제. `python eval.py policy=websocket`
- `env_wrapper._target_` — Hydra `instantiate()`로 런타임에 래퍼 클래스 생성
- 챌린지 참가팀은 `env_wrapper._target_`만 자신의 래퍼 클래스로 오버라이드
- `register_omegaconf_resolvers()` — 커스텀 OmegaConf resolver 등록

### 결과 수집

**Per-instance JSON** 출력: `{task_name}_{instance_id}_{episode}.json`

메트릭 구조:
```json
{
  "q_score": {"final": 0.75},
  "time": {"simulator_steps": 1200, "simulator_time": 60.0, "normalized_time": 0.8},
  "agent_distance": {"base": 5.2, "left": 3.1, "right": 2.8},
  "normalized_agent_distance": {"base": 1.1, "left": 0.9, "right": 0.95}
}
```

**score_utils.py** (204줄) — 최종 점수 집계:
- **q_score**: 인스턴스별 → 태스크별 평균 → 전체 평균
- **task_sr**: 태스크 성공률 (q_score == 1.0인 비율)
- **time_score**: `2 - 1/normalized_time` — 빠를수록 높은 점수 (최대 2.0, 인간 수준이 1.0)
- **distance_scores**: base/left/right 각각
- 제출 디렉토리 명명: `{track}.{testset}.{team}.{affiliation}.{date}/json/`

## 프로젝트별 심층 분석

### Evaluator 패턴 심층 분석

`Evaluator` 클래스의 설계 특징:

1. **컨텍스트 매니저**: `__enter__`에서 환경 로드 + 워밍업, `__exit__`에서 정리. SIGINT 핸들러 등록하여 Ctrl+C 시 graceful shutdown
2. **Human-stats 기반 timeout**: `max_steps`가 null이면 인간 데모 평균 완료 스텝 × 2를 자동 사용. 태스크마다 다른 시간 제한
3. **비디오 기록**: PyAV를 이용한 평가 비디오 녹화 (`_write_video`). 프레임 단위로 인코딩, `write_video` config 플래그로 제어
4. **관찰 전처리**: `_preprocess_obs`가 환경 출력을 모델 입력에 맞게 변환 — dict 평탄화, 카메라 상대 pose 계산, 태스크 ID 주입

**우리 프로젝트 EpisodeRunner와의 대응**: Evaluator는 "벤치마크 + 에피소드 러너 + 결과 수집"을 하나의 클래스에 통합. 우리 설계에서는 이를 Benchmark / EpisodeRunner / ResultCollector로 분리.

### MetricBase 콜백 시스템

```python
class MetricBase(ABC):
    @abstractmethod def start_callback(self, env): ...   # 에피소드 시작
    @abstractmethod def step_callback(self, env): ...     # 매 스텝
    @abstractmethod def end_callback(self, env): ...      # 에피소드 종료
    @abstractmethod def gather_results(self) -> dict: ... # 결과 수집
```

**구현체 2개**:
- `AgentMetric`: 매 스텝 로봇 base + 양손 end-effector 이동 거리 누적. `end_callback`에서 human baseline 대비 `normalized_agent_distance` 계산
- `TaskMetric`: `start_callback`에서 초기 predicate 상태 저장. `end_callback`에서 `q_score` 계산 — 태스크 성공 시 1.0, 아니면 max(새로 만족된 predicate 수 / 전체 predicate 수) across all groundings. `normalized_time` = agent_time / human_time

**채택 가치**: 이 콜백 패턴은 매우 깔끔하고 확장성 있음. 새 메트릭 추가 시 MetricBase만 구현하면 Evaluator가 자동으로 호출. 우리 프로젝트의 ResultCollector에도 유사한 콜백 인터페이스를 고려할 가치.

### 분산 평가 시스템 (eval-jobqueue)

**FastAPI 기반 작업 큐 서버** (`jobqueue.py`, 468줄):

| 컴포넌트 | 역할 |
|----------|------|
| `Job` | id, payload, resource_type, status (pending/in_progress/done), assigned_to, heartbeat |
| `ResourceLease` | per-allocation 추적: worker_id, job_id, idx, last_heartbeat |
| `ResourcePool` | thread-safe 리소스 할당. `BoundedSemaphore`로 동시성 제어, lease 기반 관리 |

**핵심 메커니즘**:
- **스마트 작업 선택** (`select_next_job`): 리소스 타입별 활용률이 낮은 쪽 우선, 같으면 시도 횟수 적은 쪽 우선
- **Heartbeat 감시**: 작업 heartbeat 60초 주기 스위프, 1시간 타임아웃 시 reclaim. 리소스 heartbeat 10초 주기, 20분 타임아웃
- **Persistent state**: `jobs.json` 원자적 쓰기 (`tempfile + os.replace`). 서버 재시작 시 done 상태 보존, 나머지는 pending으로 복귀
- **All-in-one heartbeat**: `POST /heartbeat?worker=X` — 해당 워커의 모든 job + resource를 한 번에 heartbeat

**분산 워커** (`eval_with_jobqueue.py`, 274줄):
```
워커 시작 → 고유 ID 생성 (user + SLURM_JOB_ID + UUID)
  → GET /job?worker=ID → 작업 수신 (team, task, instance)
  → Evaluator 생성 → POST /resource/{type}/acquire → 모델 서버 host:port 획득
  → WebsocketPolicy.update_host(host, port) → 평가 루프 (FPS 로깅)
  → POST /resource/{type}/release → POST /done/{job_id}
  → 백그라운드: 30초 간격 heartbeat 스레드
```

### Challenge Submission 워크플로우

전체 파이프라인:

1. **참가팀**: `submission.Dockerfile`로 모델을 WebSocket 정책 서버로 패키징
2. **generate_jobs.py**: 제출물 분석 → top-5 팀 선정 (q_score 기준) → (team, task, instance) 트리플 생성 → 이미 완료된 인스턴스 스킵 → `jobs.json` + `resources.json` 출력
3. **jobqueue.py**: FastAPI 서버로 작업 배분
4. **eval_with_jobqueue.py**: 워커가 작업 수신 → 팀의 모델 서버 컨테이너에 WebSocket 연결 → 팀의 커스텀 wrapper로 Evaluator 생성 → 평가 실행 → 결과 JSON 저장

**Per-team wrapper**: `challenge_submissions/submission_{team_slug}.py` — 각 팀은 `EnvironmentWrapper` 서브클래스로 관찰 전처리를 커스터마이징 (카메라 해상도, modality, proprioception 포맷 등).

## 코드 품질/성숙도

| 항목 | 평가 |
|------|------|
| 타입 힌트 | 부분적. 핵심 인터페이스(eval.py, policies.py)에는 있으나 일관성 부족 |
| 테스트 | 평가 파이프라인 자체에 대한 유닛 테스트 없음. 시뮬레이션 의존성이 높아 테스트가 어려운 구조 |
| 문서화 | docstring 최소한. 코드 내 주석은 적절. README 존재하지만 평가 인프라 설명은 부족 |
| CI/CD | Docker 빌드 기반. 자동화된 테스트 파이프라인 미확인 |
| 코드 구조 | learning/ 디렉토리가 깔끔하게 분리됨. eval.py가 단일 파일에 너무 많은 책임 (497줄) |
| 성숙도 | 실제 챌린지 운영에 사용된 프로덕션 코드. eval-jobqueue는 실전 검증됨 |

## 시사점

### 채택할 패턴

1. **Evaluator 콜백 → Benchmark ABC의 `is_done()` + `get_result()`**: OmniGibson의 `_create_termination_conditions()`/`_create_reward_functions()` 콜백 패턴은 우리 Benchmark ABC의 `is_done(step_result)` 및 `get_result(step_result)` 메서드에 대응. 종료 조건과 결과 판정을 벤치마크 구현자에게 위임하는 동일한 설계 원칙
2. **Task JSON config → YAML config + Benchmark.get_tasks()**: BEHAVIOR-1K의 태스크 인스턴스 목록(train/test/hidden)과 per-instance 설정은 우리 YAML config의 `tasks` 필드와 `Benchmark.get_tasks()` 반환값에 대응. 태스크 목록을 선언적으로 관리하는 패턴
3. **Docker 격리 전략 → per-benchmark Docker 이미지 (Orchestrator 책임)**: Submission Docker(모델 서버 컨테이너)와 메인 Docker(시뮬레이터 컨테이너)의 이중 구조는 우리 설계의 per-benchmark Docker 이미지 + 외부 ModelServer 분리와 부합. Orchestrator가 컨테이너 기동/종료를 관리
4. **Evaluator 에피소드 루프 → EpisodeRunner 패턴**: `reset() → while not done: step()` 루프는 우리 SyncEpisodeRunner의 `run_episode()` 구조와 직접 대응. 다만 OmniGibson은 이를 Evaluator 단일 클래스에 통합한 반면, 우리는 EpisodeRunner로 분리
5. **MetricBase 콜백 + score_utils 집계 → Result Collector**: `start_callback/step_callback/end_callback/gather_results` 콜백과 per-instance → per-task → overall 집계 구조는 우리 Result Collector의 에피소드/태스크/벤치마크 단위 메트릭 수집 및 집계에 대응
6. **Health check + graceful 연결**: WebSocket 연결 전 `/healthz` 체크, reconnection 지원 — 우리 Orchestrator의 서버 연결 확인 단계에 적용할 패턴
7. **서버 타이밍 메타데이터**: 응답에 `infer_ms`, `prev_total_ms` 포함 — 우리 Protocol 메시지의 `timestamp` 필드와 Result Collector의 추론 지연 시간 메트릭에 활용 가능

### 회피할 패턴

1. **단일 파일 과부하**: `eval.py` (497줄)가 환경 로드, 정책 관리, 관찰 전처리, 비디오 기록, 메인 루프를 모두 포함 — 우리는 이를 Benchmark / EpisodeRunner / Result Collector / Orchestrator로 분리
2. **글로벌 상태**: `jobqueue.py`가 모듈 레벨 변수 (`jobs`, `job_index`, `pending_jobs` 등)를 사용 — 테스트 어렵고 서버 다중 인스턴스 불가
3. **하드코딩된 상수**: `eval_utils.py`의 `ROBOT_CAMERA_NAMES`, `PROPRIOCEPTION_INDICES` 등이 모두 Python dict 리터럴 — 우리 YAML config 시스템으로 외부화하는 것이 유지보수에 유리
4. **단일 벤치마크 종속**: 모든 코드가 BEHAVIOR-1K에 긴밀히 결합 — 우리 Benchmark ABC는 이를 추상화하여 다중 벤치마크 지원을 구조적으로 보장

### 열린 질문

1. **메트릭 콜백의 데이터 범위**: OmniGibson의 MetricBase는 env를 직접 참조하여 내부 상태를 읽음. 우리 설계에서는 환경이 Docker 컨테이너 안에 있으므로, Result Collector가 수집하는 메트릭은 `Benchmark.get_result()`가 반환하는 dict와 Protocol 메시지의 타이밍 정보만으로 충분한지, 아니면 Benchmark ABC에 추가 메트릭 인터페이스가 필요한지?
2. **분산 평가 확장**: eval-jobqueue의 FastAPI 작업 큐 + heartbeat 패턴은 대규모 평가에 유용. 현재 Orchestrator는 단일 머신 순차 실행이지만, 향후 multi-node 확장 시 유사한 작업 큐 패턴을 Orchestrator 위에 구축할 수 있는지?
3. **관찰 전처리 위치**: OmniGibson은 팀별 EnvironmentWrapper로 관찰 전처리를 커스터마이징. 우리 설계에서는 `Benchmark.make_obs()`가 이 역할을 담당하는데, 모델별로 다른 관찰 포맷이 필요한 경우 make_obs()의 유연성이 충분한지?
