# OpenEnv — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/meta-pytorch/OpenEnv |
| Last Commit | 2026-02-18 (매우 활발) |
| 총 Python 파일 | 516개 (src/ 약 14,000줄 + envs/ 29개 환경) |
| 주요 목적 | LLM 에이전트용 격리 실행 환경의 생성·배포·사용 e2e 프레임워크 (Gymnasium-style API + Docker + MCP) |
| 조직 | Meta (FAIR) |
| 우리 프로젝트에서의 참고 포인트 | Gymnasium-style API 위에 WebSocket과 Docker 격리를 결합한 client-server 아키텍처 설계 |

## 프로젝트 구조

```
OpenEnv/
├── src/openenv/
│   ├── core/
│   │   ├── env_client.py          # EnvClient ABC — WebSocket 기반 비동기 클라이언트
│   │   ├── sync_client.py         # SyncEnvClient — 동기 래퍼
│   │   ├── client_types.py        # StepResult[ObsT] 데이터클래스
│   │   ├── mcp_client.py          # MCPToolClient — MCP 도구 호출 클라이언트
│   │   ├── env_server/
│   │   │   ├── interfaces.py      # Environment ABC, Transform ABC, Rubric 통합
│   │   │   ├── http_server.py     # HTTPEnvServer — FastAPI + WebSocket 서버 래퍼
│   │   │   ├── types.py           # Action, Observation, State (Pydantic BaseModel)
│   │   │   ├── serialization.py   # 관찰/액션 직렬화
│   │   │   ├── mcp_types.py       # JSON-RPC MCP 타입
│   │   │   └── route_config.py    # 엔드포인트 등록 설정
│   │   ├── containers/
│   │   │   └── runtime/
│   │   │       ├── providers.py   # ContainerProvider, RuntimeProvider ABC
│   │   │       ├── uv_provider.py # UVProvider (로컬 실행)
│   │   │       └── daytona_provider.py
│   │   ├── rubrics/               # 보상 계산 시스템
│   │   │   ├── base.py            # Rubric ABC (nn.Module 패턴)
│   │   │   ├── containers.py      # 복합 루브릭
│   │   │   ├── llm_judge.py       # LLM 기반 보상
│   │   │   └── trajectory.py      # 궤적 기반 보상
│   │   └── tools/                 # MCP 도구 관련
│   ├── auto/
│   │   └── _discovery.py          # 환경 자동 발견 (importlib.metadata 기반)
│   └── cli/
│       └── commands/              # init, push, build, serve, fork, validate
├── envs/                          # 29개 예시 환경
│   ├── echo_env/                  # 기본 테스트 환경
│   ├── coding_env/                # 코드 실행 환경
│   ├── dm_control_env/            # MuJoCo 연속 제어 (로봇/물리 시뮬레이션!)
│   ├── browsergym_env/            # 브라우저 자동화
│   ├── atari_env/                 # Atari 게임
│   └── ...                        # chess, calendar, git, unity, wildfire 등
└── rfcs/
    ├── 001-abstractions.md        # 핵심 추상화 RFC
    ├── 002-env-spec.md            # 프레임워크 스펙 RFC
    └── ...
```

## 핵심 분석

### 벤치마크 추상화

**서버 사이드** — `Environment(ABC, Generic[ActT, ObsT, StateT])`:
- `reset(seed, episode_id, **kwargs) → ObsT` — 에피소드 초기화
- `step(action: ActT, timeout_s, **kwargs) → ObsT` — 액션 실행
- `state → StateT` (property) — 현재 상태 조회
- `reset_async()`, `step_async()` — 비동기 변형 (기본: sync 호출 위임)
- `_apply_rubric()`, `_apply_transform()` — 보상/관찰 변환 훅

**클라이언트 사이드** — `EnvClient(ABC, Generic[ActT, ObsT, StateT])`:
- 동일한 `reset()`, `step()`, `state()` 메서드 (WebSocket 경유)
- 추상 메서드: `_step_payload()`, `_parse_result()`, `_parse_state()` — 직렬화/역직렬화
- Factory: `from_docker_image()`, `from_env()` (HF Space 지원)

**기본 타입** — Pydantic `BaseModel` 기반:
- `Action`: `metadata: Dict[str, Any]`, `extra="forbid"`
- `Observation`: `done: bool`, `reward: float|None`, `metadata: Dict`
- `State`: `episode_id`, `step_count`
- `StepResult[ObsT]`: `observation`, `reward`, `done` (클라이언트 측)

**핵심 차이점**: OpenEnv는 LLM 에이전트 환경(텍스트 기반 액션)이 주 대상. 하지만 `dm_control_env`가 연속 제어(float 리스트 액션, 물리 시뮬레이션)를 이미 지원하므로, Gymnasium-style API가 로봇 제어에도 적용 가능함을 보여줌.

### 모델 통합 패턴

**"Thin Agent" 원칙** — OpenEnv는 모델 통합을 **의도적으로 프레임워크 밖으로** 둔다:
- `Agent` ABC는 RFC 001에서 정의되나 실제 구현은 최소한 (act → action)
- `ModelTokenizer` Protocol: HuggingFace 호환 토크나이저 인터페이스만 정의
- MCP 인터페이스가 에이전트-환경 상호작용의 주 채널

우리 프로젝트에서는 `ModelServer` 계층이 핵심이므로 이 부분은 직접 참고할 것이 적음. 다만 "프레임워크가 모델을 직접 감싸지 않는다"는 설계 결정의 근거를 참고할 수 있음.

### 에피소드 실행 루프

**내장 EpisodeRunner 없음** — RFC 001에서 보여주는 루프:
```python
result = await env.reset()
while not result.done:
    action = agent.act(result.observation)
    result = await env.step(action)
```

훈련 인프라(RL 코드)가 루프를 소유. OpenEnv는 환경 인터페이스만 제공. PyTorch `IterableDataset`으로 TaskDataset을 제공하여 DataLoader 통합 지원.

우리 프로젝트의 `EpisodeRunner` (Sync/Async 분리) 패턴은 OpenEnv에 없는 고유 추상화.

### 통신 구조

**이중 프로토콜 설계**:
1. **WebSocket `/ws`** — 시뮬레이션 제어 (reset/step/state/close/mcp 메시지 타입)
   - 세션당 전용 Environment 인스턴스
   - JSON 텍스트 프레임 (`json.dumps` / `json.loads`)
   - 메시지 구조: `{"type": "step", "data": {...}}`
2. **WebSocket `/mcp`** — MCP JSON-RPC (에이전트 도구 호출)
   - JSON-RPC 2.0 프로토콜 (`tools/list`, `tools/call`)
3. **HTTP REST** — 상태 비저장 연산 (`/reset`, `/step`, `/state`, `/health`, `/schema`, `/metadata`)

**직렬화**: JSON only (`json.dumps`/`json.loads`). 바이너리 프로토콜(msgpack 등) 미사용.

**우리 설계와의 비교**: 우리는 WebSocket + msgpack (바이너리)을 채택. 이미지(480×640×3 = ~1MB)를 주고받는 로봇 제어에서 JSON은 비효율적. OpenEnv도 `max_message_size_mb=100`으로 큰 메시지를 허용하지만 텍스트 인코딩 오버헤드가 있음.

### 환경 격리

**Docker 컨테이너 격리** — 환경당 독립 컨테이너:
- `ContainerProvider` ABC: `start_container(image)→url`, `stop_container()`, `wait_for_ready()`
- 구현체: `LocalDockerProvider`, `DockerSwarmProvider`, `KubernetesProvider`(stub), `UVProvider`(비컨테이너)
- `from_docker_image()` factory → 이미지 pull → 컨테이너 시작 → health check → 클라이언트 연결
- `from_env()` factory → HuggingFace Spaces 지원 (Docker/UV 모드)

**세션 관리** (HTTPEnvServer):
- WebSocket 연결당 전용 Environment 인스턴스 생성
- `_sessions: Dict[str, Environment]` — 세션 ID → 환경 매핑
- `SessionInfo`: 생성 시각, 마지막 활동, 스텝 카운트
- `SUPPORTS_CONCURRENT_SESSIONS` 클래스 플래그로 동시 세션 안전성 선언
- `ConcurrencyConfig`: 최대 동시 환경 수, 세션 타임아웃
- `ThreadPoolExecutor` per session — sync 환경을 async 컨텍스트에서 실행

### 설정/CLI

**CLI 명령**:
- `openenv init <env_name>` — 새 환경 스캐폴딩 (템플릿 기반)
- `openenv push` — HuggingFace Spaces 배포
- `openenv build` — Docker 이미지 빌드
- `openenv serve` — 로컬 서버 시작
- `openenv fork` — 환경 포크
- `openenv validate` — 환경 검증

**설정 체계**: `openenv.yaml` 매니페스트 파일 (환경당 1개):
```yaml
spec_version: 1
name: echo_env
description: "Echo Environment"
action: EchoAction
observation: EchoObservation
```

lm-eval-harness의 복잡한 YAML 태스크 정의와 달리 매우 단순. Convention over Configuration 원칙 — 대부분 클래스명을 환경명에서 추론 (`echo_env` → `EchoEnv`, `EchoAction`, `EchoObservation`).

### 결과 수집

**최소한의 결과 관리**:
- `StepResult(observation, reward, done)` — 스텝 단위 결과
- `Observation.reward` — 환경 내부에서 계산된 보상
- Rubric 시스템으로 보상 계산 구조화 가능
- **에피소드/실험 단위 결과 집계, 로깅, 영속화 시스템 없음**

OpenEnv는 "환경 제공"이 목적이며, 결과 집계는 상위 훈련 인프라의 책임으로 명확히 분리.

## 프로젝트별 심층 분석

### Client-Server 분리 패턴

OpenEnv의 가장 핵심적인 아키텍처 결정. 환경(서버)과 에이전트(클라이언트)를 프로세스·네트워크 경계로 완전 분리:

**서버** — `HTTPEnvServer`:
- `env: Callable[[], Environment]` factory 패턴 — 세션마다 새 Environment 인스턴스 생성
- `_create_session(websocket)` / `_destroy_session(session_id)` — async lock으로 동시 접근 보호
- `_sessions: Dict[str, Environment]`, `_session_executors: Dict[str, ThreadPoolExecutor]`, `_session_info: Dict[str, SessionInfo]`
- Sync Environment를 `loop.run_in_executor()`로 감싸 비동기 서버에서 실행

**클라이언트** — `EnvClient`:
- WebSocket 전용 통신. 서버의 내부 구현(Environment 클래스, 상태 관리)을 전혀 모름
- async-by-default. `.sync()` 호출로 `SyncEnvClient` 래퍼 반환 (이벤트 루프 자동 관리)
- 추상 메서드 `_step_payload()`, `_parse_result()`, `_parse_state()`로 직렬화/역직렬화를 서브클래스에 위임
- `from_docker_image()` / `from_env()` factory classmethod — 인프라 설정까지 원스텝

**우리 설계에 대한 시사점**: 우리의 Benchmark(서버) ↔ ModelServer(클라이언트) 분리와 구조적으로 동일. 다만 우리는 **모델 쪽이 서버**이고 **벤치마크 쪽이 클라이언트**인 점이 반대. OpenEnv의 per-session Environment 인스턴스 + factory 패턴은 우리의 per-connection 세션 관리에 직접 적용 가능.

### Dual-mode 아키텍처 (Simulation vs Production)

```python
class ServerMode(str, Enum):
    SIMULATION = "simulation"   # reset/step/state + /ws
    PRODUCTION = "production"   # health/schema/metadata + /mcp only
```

**Simulation 모드**: `/reset`, `/step`, `/state` HTTP 엔드포인트 + `/ws` WebSocket 등록. 훈련·평가용.

**Production 모드**: `/health`, `/schema`, `/metadata` HTTP 엔드포인트 + `/mcp` WebSocket만 등록. 시뮬레이션 제어 엔드포인트 제거. 에이전트가 MCP 도구 호출로만 상호작용.

`register_routes(app, mode)` 함수가 모드에 따라 엔드포인트를 조건부 등록. 클라이언트의 `_mode` 속성은 생성 후 불변 (`OPENENV_CLIENT_MODE` 환경변수 또는 생성자 파라미터).

**Graceful Degradation**: 동일한 환경 코드가 훈련(시뮬레이션)에서 배포(프로덕션)로 전환할 때 시뮬레이션 제어 레이어만 제거. 환경 코드 자체는 변경 없음.

**우리 설계에 대한 시사점**: 우리의 Sync/Async EpisodeRunner 분리와 개념적으로 유사. "동일한 Benchmark 구현이 실행 모드에 따라 다르게 동작한다"는 원칙 채택 가능. 다만 우리 맥락에서는 sim/prod이 아닌 sync(시뮬레이션 시간)/async(실시간) 평가 모드.

### The Time Problem

RFC 001에서 명시적으로 프레이밍한 핵심 문제:

> **Simulation time**: 에이전트가 행동할 때까지 시간이 멈춤. Step-based. 에이전트의 추론 시간이 환경 시간에 영향 없음.
>
> **Real time**: 시간이 연속적으로 흐름. 멈출 수 없음. 에이전트가 느리면 환경이 먼저 진행.

OpenEnv의 step-based API는 **암묵적으로 simulation time**을 사용. 내장 real-time 실행 모드 없음. RFC에서 이 문제를 인식하고 있으나 해결은 future work으로 남김.

**우리 설계에 대한 시사점**: 이것이 정확히 우리 `AsyncEpisodeRunner`의 존재 이유. OpenEnv가 해결하지 않은 문제를 우리가 해결해야 함. "환경 시간이 에이전트 추론과 독립적으로 흐르는" 비동기 평가 모드는 우리 프로젝트의 핵심 차별점.

### Auto-discovery 시스템

`EnvironmentDiscovery` 클래스 — 설치된 `openenv-*` 패키지를 자동 발견:
- `importlib.metadata.distributions()`으로 설치된 패키지 스캔
- 패키지 리소스에서 `openenv.yaml` 매니페스트 로드
- Convention-based 클래스명 추론: `echo_env` → `EchoEnv`, `EchoAction`, `EchoObservation`
- `EnvironmentInfo` 데이터클래스: 동적 import 메서드 (`get_client_class()`, `get_action_class()`)
- 글로벌 싱글톤 + 메모리 캐시 + 파일 캐시 (`/tmp/openenv_discovery_cache.json`)

**우리 설계에 대한 시사점**: lm-eval-harness의 Registry와 유사한 목적이지만 접근 방식이 다름. lm-eval-harness는 데코레이터 기반 등록, OpenEnv는 패키지 메타데이터 기반 자동 발견. 우리 프로젝트는 벤치마크 수가 제한적이므로 lm-eval-harness 스타일의 명시적 레지스트리가 더 적합할 수 있으나, convention-over-configuration 원칙은 참고할 가치 있음.

### Rubric/Transform 시스템

**Rubric** — `nn.Module`을 모델링한 보상 계산 프레임워크:
```python
class Rubric(ABC):
    def __init__(self):
        self._rubric_children = {}      # 자식 루브릭 자동 등록
        self._forward_hooks = []         # post-forward 훅
        self._forward_pre_hooks = []     # pre-forward 훅
        self.last_score = None           # 최근 점수 추적

    def __setattr__(self, name, value):
        if isinstance(value, Rubric):
            self._rubric_children[name] = value  # 자동 등록
        object.__setattr__(self, name, value)

    @abstractmethod
    def forward(self, action, observation) -> float: ...

    def __call__(self, action, observation):  # sync/async 자동 분기
```

- `containers.py`: 복합 루브릭 (여러 루브릭 조합)
- `llm_judge.py`: LLM 기반 보상 계산
- `trajectory.py`: 궤적 전체에 대한 보상
- Environment의 `step()` 내부에서 `_apply_rubric(action, obs)` 자동 호출, `reset()`에서 `_reset_rubric()` 호출

**Transform** — TorchRL 영감의 관찰 변환:
- `Transform(ABC, Generic[ObsT])` — 관찰을 파이프라인으로 변환
- `__setattr__` 기반 자식 Transform 자동 등록 (Rubric과 동일 패턴)

**우리 설계에 대한 시사점**: Rubric의 `nn.Module` 패턴은 우리의 메트릭 계산에 영감을 줄 수 있음. 특히 계층적 메트릭 구성(성공률 → 서브태스크 성공률 → 안전성 메트릭 등)에 `__setattr__` 기반 자동 등록과 `named_rubrics()` 인트로스펙션이 유용. 다만 우리는 step-level 보상보다 episode-level 메트릭이 주이므로 직접 적용보다는 패턴만 차용.

### Container Provider 추상화

두 개의 ABC 계층:

**ContainerProvider** (Docker 기반):
- `start_container(image, port, env_vars, ...) → url`
- `stop_container()`
- `wait_for_ready(url, timeout, interval)` — `/health` 엔드포인트 폴링
- 구현: `LocalDockerProvider` (subprocess `docker run`, 포트 자동 할당, health 폴링), `DockerSwarmProvider` (replicated service, overlay network, 로드밸런서), `KubernetesProvider` (stub)

**RuntimeProvider** (비컨테이너):
- `start()`, `stop()`, `wait_for_ready()`
- 구현: `UVProvider` (로컬 `uv run` 실행)

**공통 패턴**: Health check via `/health` 엔드포인트 폴링, 설정 가능한 timeout/interval.

**우리 설계에 대한 시사점**: 우리 `05_DESIGN_OUTLINE.md`에서도 Docker 기반 환경 격리를 명시. OpenEnv의 ContainerProvider ABC 구조(특히 `LocalDockerProvider`의 포트 관리, health check 로직)를 거의 그대로 채택 가능. `SUPPORTS_CONCURRENT_SESSIONS` 플래그로 환경별 동시 세션 안전성을 선언하는 패턴도 유용.


## 코드 품질/성숙도

| 항목 | 평가 |
|------|------|
| 타입 힌트 | ◎ 우수 — Pydantic BaseModel + Generic[ActT, ObsT, StateT] 전면 사용. 런타임 검증까지 제공 |
| 테스트 | △ 제한적 — src/ 내에 가시적인 테스트 디렉토리 부재. envs/ 내 개별 환경에 일부 테스트 존재 |
| CI/CD | ○ 양호 — GitHub Actions 기반 (HuggingFace Spaces 배포 파이프라인 포함) |
| 문서화 | ◎ 우수 — RFC 기반 설계 프로세스, 모든 public 클래스에 상세 docstring, 29개 예시 환경 |
| 코드 구조 | ◎ 우수 — core/client/server/containers/rubrics 명확한 모듈 분리. 순환 의존성 없음 |
| 성숙도 | ○ 초기-중기 — Meta FAIR 프로젝트, BSD 라이선스, 활발한 개발 중 (last commit 3일 전) |

**특기사항**: RFC-driven 설계 프로세스가 인상적. 001-abstractions.md (570줄), 002-env-spec.md (439줄)에서 핵심 설계 결정의 근거를 상세히 기록. 이는 우리 프로젝트의 docs/ 디렉토리 구조(PROPOSAL → LANDSCAPE → PHILOSOPHY → OUTLINE)와 정신적으로 유사.

## 시사점

### 채택할 패턴

1. **Client-Server WebSocket 분리 → Protocol + Connection + ModelServer**: 환경(서버)과 모델(클라이언트)을 네트워크 경계로 완전 분리하는 패턴. 우리 설계의 Protocol(WebSocket + msgpack) + Connection(클라이언트 라이브러리) + ModelServer(서버 ABC) 구조와 직접 대응. 단, **방향이 반대**: OpenEnv는 환경이 서버·에이전트가 클라이언트이지만, 우리는 모델이 서버·벤치마크가 클라이언트.

2. **Per-session Environment + Factory → Benchmark ABC의 `reset()`**: `env: Callable[[], Environment]`로 세션마다 새 인스턴스 생성하여 상태 격리 보장. 우리 설계에서는 `Benchmark.reset(task)`가 태스크마다 새 환경 인스턴스를 생성·반환하여 동일한 격리를 달성.

3. **`SUPPORTS_CONCURRENT_SESSIONS` 안전성 플래그 → `BatchPredictModelServer` 멀티세션**: 환경 구현체가 동시 세션 지원 여부를 클래스 레벨에서 선언. 우리 설계에서 `BatchPredictModelServer`가 여러 session의 observation을 동시에 배치 추론하는 시나리오에서 유사한 안전성 선언이 필요. Benchmark 측에서도 동시 에피소드 실행 가능 여부를 선언하는 플래그 고려 가능.

4. **ContainerProvider ABC → Orchestrator의 Docker 관리**: Docker/Swarm/K8s를 추상화한 컨테이너 관리 인터페이스. 우리 설계에서는 `Orchestrator`가 벤치마크별 Docker 컨테이너 기동·종료·health check를 담당. OpenEnv의 `LocalDockerProvider`(포트 할당, health check 폴링)를 Orchestrator 내부 구현의 참고로 활용.

5. **Dual-mode 아키텍처 → SyncEpisodeRunner / AsyncEpisodeRunner**: 동일 코드가 모드에 따라 다른 인터페이스를 노출하는 개념. 우리 설계에서는 동일한 `Benchmark` 구현이 `SyncEpisodeRunner`(시뮬레이션 시간 — 추론 완료까지 대기)와 `AsyncEpisodeRunner`(실시간 — wall-clock Hz + hold_policy)에 모두 조합됨. 벤치마크 코드 수정 없이 실행 모드만 전환.

6. **The Time Problem → AsyncEpisodeRunner의 존재 이유**: OpenEnv RFC 001이 명시적으로 프레이밍한 simulation time vs real time 문제. OpenEnv는 이를 future work으로 남겼으나, 우리 설계에서는 `AsyncEpisodeRunner`가 정확히 이 문제를 해결 — wall-clock 기반 고정 Hz로 환경을 step하고, 추론이 완료되지 않으면 `hold_policy`(`repeat_last`, `zero`, callable)를 적용.

7. **Auto-discovery → 데코레이터 기반 Registry**: OpenEnv는 `importlib.metadata` 기반 패키지 자동 발견. 우리 설계에서는 lm-eval-harness 스타일의 데코레이터 기반 명시적 Registry를 채택 (벤치마크 수가 제한적이므로 더 적합). Convention-over-configuration 원칙은 YAML 설정의 벤치마크명 → Benchmark 클래스 매핑에 적용.

8. **Rubric/Transform → Result Collector + Benchmark.make_obs()**: Rubric의 계층적 메트릭 구성 패턴은 우리 `Result Collector`의 에피소드/태스크/벤치마크 단위 메트릭 집계에 영감. Transform의 관찰 변환 파이프라인은 우리 `Benchmark.make_obs()`가 담당 — 환경 raw observation을 모델 서버에 전송할 dict로 변환.

9. **Immutable mode after init → SessionContext**: 클라이언트 모드가 생성 후 변경 불가. 우리 `SessionContext`의 `mode` 속성(`"sync"` | `"realtime"`)도 에피소드 시작 시 결정되어 변경 불가.

10. **Environment.reset/step → Benchmark.reset/step**: OpenEnv의 `Environment.reset()` → `ObsT`, `Environment.step(action)` → `ObsT` 패턴이 우리 `Benchmark.reset(task)` → `(env, obs_dict)`, `Benchmark.step(env, action)` → `StepResult`에 직접 대응.

11. **StepResult → StepResult**: OpenEnv의 `StepResult[ObsT](observation, reward, done)`이 우리 `StepResult(obs, reward, done, info)` 데이터클래스에 직접 대응.

### 회피할 패턴

1. **JSON 텍스트 직렬화 → msgpack 바이너리 (Protocol)**: 이미지 전송에 비효율적. base64 인코딩 → JSON 문자열화 → 디코딩 과정의 오버헤드. 우리 Protocol은 WebSocket + msgpack + msgpack-numpy로 numpy array를 포함한 임의의 dict를 바이너리 직렬화.

2. **MCP 의존**: Model Context Protocol은 LLM 에이전트의 도구 호출에 특화. VLA 모델의 연속 제어 액션(float 배열)에는 불필요한 추상화 레이어. 우리에게 직접적으로 불필요.

3. **내장 결과 집계 부재 → Result Collector**: OpenEnv는 환경 제공이 목적이므로 의도된 결정이지만, 우리에게는 에피소드/태스크/벤치마크 단위 결과 집계가 필수. `Result Collector`가 에피소드별 메트릭 수집, 집계, reference score 비교, 리포팅을 담당.

4. **Pydantic 기반 액션/관찰 모델 → `dict[str, Any]` + numpy**: `Action`/`Observation`이 Pydantic BaseModel이지만 주로 텍스트/구조화 데이터 대상. 우리는 `dict[str, Any]`에 numpy array를 직접 담아 전송 — Pydantic 모델의 직렬화/검증 오버헤드 없이 고차원 연속 액션 공간(7-DoF 로봇 팔) + 이미지 관찰을 효율적으로 처리.

5. **에이전트가 reset을 트리거하지 않는 원칙**: OpenEnv RFC에서 명시. 우리 컨텍스트에서는 `EpisodeRunner`가 `Benchmark.reset()`을 호출하여 에피소드 초기화를 관리하므로 다른 방식으로 해결됨.

### 열린 질문

1. **OpenEnv의 dual-mode(sim/prod)를 우리의 sync/async 평가 모드에 어떻게 매핑할 것인가?** → **부분적으로 해결됨.** Simulation mode ≈ 우리 `SyncEpisodeRunner` (step-based, 추론 완료까지 환경 시간 멈춤). 그러나 OpenEnv의 Production mode는 MCP 도구 호출 기반이므로 우리 `AsyncEpisodeRunner`(wall-clock Hz + hold_policy)와는 직접 대응하지 않음. 우리의 sync/async 구분은 OpenEnv의 sim/prod과 다른 축 — "시간 모델"의 차이이지 "인터페이스 노출 범위"의 차이가 아님.

2. **Container provider 추상화를 어느 수준까지 채택할 것인가?** → **우리 Orchestrator는 LocalDocker로 시작하되 확장 가능한 인터페이스를 유지.** OpenEnv의 `ContainerProvider` ABC 구조를 참고하여, Orchestrator 내부에 Docker 관리 로직을 추상화. 초기에는 로컬 Docker만 지원하지만, 인터페이스 변경 없이 Swarm/K8s 등으로 확장 가능한 구조.

3. **Rubric 패턴을 우리의 메트릭 계산에 적용할 수 있는가?** `nn.Module`-like 계층적 메트릭 구성이 벤치마크별 메트릭 정의에 유용할 수 있지만, 오버엔지니어링 위험도 있음. 우리 `Result Collector`는 에피소드/태스크/벤치마크 3단계 집계에 집중하며, Rubric의 훅 시스템보다는 단순한 메트릭 딕셔너리 기반 접근이 적합할 수 있음.

4. **dm_control_env의 연속 액션 처리 방식이 VLA 환경에 적용 가능한가?** → **우리 설계로 해결됨.** `DMControlAction(values: List[float])`의 Pydantic 모델 방식 대신, 우리는 `dict[str, Any]`에 numpy array를 직접 담아 전송 (예: `{"actions": np.array([0.1, -0.2, ...])}`). Pydantic 모델의 직렬화/검증 오버헤드 없이 고차원 연속 액션을 효율적으로 처리.

5. **OpenEnv의 "환경이 보상을 계산한다" 원칙을 우리도 따를 것인가?** 시뮬레이션 환경(LIBERO, SimplerEnv 등)이 자체 성공 판정을 제공하는 경우가 대부분이므로 자연스러움. 우리 `Benchmark.step()`이 `StepResult(reward=...)`로 환경 보상을 반환하고, `Benchmark.get_result()`가 에피소드 최종 결과를 산출. 추가 메트릭(smoothness, safety 등)은 `Result Collector`에서 외부 계산.
