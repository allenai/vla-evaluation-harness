# starVLA — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/starVLA/starVLA |
| Stars / Last Commit | 활발한 개발 중 / 2025-02-21 (서베이 당일) |
| 규모 | Python 95개 파일, ~24,900 라인 |
| 라이선스 | MIT License |
| 주요 목적 | "Lego-like Codebase for Vision-Language-Action Model Developing" — 모듈형 VLA 모델 개발 프레임워크. 학습과 평가를 모두 지원하며 다수 벤치마크 통합 |
| 원류 | InternVLA-M1에서 fork |
| 우리 프로젝트에서의 참고 포인트 | 01_PROPOSAL.md: "다양한 벤치마크(LIBERO, SimplerEnv, BEHAVIOR-1K, Calvin 등)와의 실질적 통합 경험" |

## 프로젝트 구조

```
starVLA/
├── starVLA/                            # 핵심 패키지
│   ├── model/
│   │   ├── framework/                  # VLA 프레임워크 구현체
│   │   │   ├── base_framework.py       # baseframework(PreTrainedModel) ABC
│   │   │   ├── QwenGR00T.py            # Qwen-VL + Flow-Matching head
│   │   │   ├── QwenFast.py             # Qwen-VL + Fast tokenizer
│   │   │   ├── QwenOFT.py              # Qwen-VL + MLP action head
│   │   │   ├── QwenPI.py               # Qwen-VL + Flow-Matching (π₀-style)
│   │   │   ├── M1.py                   # InternVLA-M1 포트
│   │   │   ├── __init__.py             # FRAMEWORK_REGISTRY + build_framework()
│   │   │   └── share_tools.py          # Config 유틸리티
│   │   ├── modules/
│   │   │   ├── action_model/           # Action heads (GR00T, Fast, OFT 등)
│   │   │   ├── dino_model/             # DINO 인코더
│   │   │   ├── projector/              # Feature projectors
│   │   │   └── vlm/                    # VLM 인터페이스 (Qwen, InternVL)
│   │   └── tools.py                    # Registry 클래스, auto_get_trainable_modules, read_mode_config
│   ├── dataloader/                     # 데이터셋 로딩 (LeRobot, QwenVL 형식)
│   ├── training/                       # 학습 스크립트
│   └── config/                         # DeepSpeed / 학습 설정
├── deployment/
│   └── model_server/
│       ├── server_policy.py            # WebSocket 정책 서버 진입점
│       └── tools/
│           ├── websocket_policy_server.py   # WebsocketPolicyServer (asyncio)
│           ├── websocket_policy_client.py   # WebsocketClientPolicy (sync)
│           ├── msgpack_numpy.py             # msgpack + numpy 직렬화
│           └── image_tools.py               # 이미지 변환 유틸
├── examples/                           # 벤치마크별 평가 코드
│   ├── LIBERO/eval_files/              # eval_libero.py + model2libero_interface.py
│   ├── SimplerEnv/eval_files/          # model2simpler_interface.py + adaptive_ensemble.py
│   ├── Behavior/                       # model2behavior_interface.py + start_behavior_env.py
│   ├── Robocasa_tabletop/              # 평가 + 학습 파일
│   ├── Robotwin/                       # 평가 + 학습 파일
│   ├── calvin/                         # 평가 + 학습 파일
│   └── eval_protocol.md                # 평가 아키텍처 문서
└── scripts/                            # 실행 스크립트
```

## 핵심 분석

### 벤치마크 추상화

**공통 Benchmark ABC 없음.** 각 벤치마크는 `examples/` 내 별도의 ad-hoc 평가 스크립트로 구현.

각 벤치마크마다 `model2{bench}_interface.py` 브릿지 파일 존재:
- `model2libero_interface.py` — LIBERO 7-DoF 액션 포맷 (`world_vector`, `rotation_delta`, `open_gripper`)
- `model2behavior_interface.py` — BEHAVIOR 23-DoF R1Pro 포맷 (base/torso/arms/grippers)
- `model2simpler_interface.py` — SimplerEnv 연동

이 인터페이스 파일들의 공통 패턴:
1. WebSocket 클라이언트 연결
2. 액션 비정규화 (un-normalization)
3. 액션 청킹 관리
4. 액션 앙상블링
5. 이미지 리사이징
6. 벤치마크 고유 액션 포맷 변환

**문제: 파일 간 상당한 코드 중복** — `unnormalize_actions`, `_check_unnorm_key`, 앙상블 로직이 파일마다 복사됨. 이는 정확히 우리 프레임워크가 해결해야 할 문제.

### 모델 통합 패턴

**baseframework(PreTrainedModel)** — HuggingFace의 PreTrainedModel을 확장한 베이스 클래스.

```python
# starVLA/model/tools.py
class Registry:
    def __init__(self, name: str):
        self._registry = {}
    def register(self, key: str):  # 데코레이터
        def decorator(framework_class):
            self._registry[key] = framework_class
            return framework_class
        return decorator

FRAMEWORK_REGISTRY = Registry("frameworks")
```

```python
# starVLA/model/framework/QwenGR00T.py
@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    def forward(self, examples): ...      # 학습용
    def predict_action(self, examples):   # 추론용
        return {"normalized_actions": np.ndarray}  # [B, T, action_dim]
```

**핵심 인터페이스 계약:**
- `forward(examples)` → 학습 시 loss 계산
- `predict_action(examples)` → 추론 시 `{"normalized_actions": np.ndarray}` 반환, shape `[B, T, action_dim]`
- `from_pretrained(ckpt_path)` → config.yaml + dataset_statistics.json + 가중치를 한 번에 로드

**자동 등록:** `pkgutil.iter_modules`로 `framework/` 하위 모듈을 자동 임포트하여 등록 트리거.

### 에피소드 실행 루프

**공통 에피소드 러너 없음.** 각 벤치마크가 자체 평가 루프 보유.

**LIBERO 예시** (`eval_libero.py`):
```python
for task in tasks:
    for episode in range(num_episodes):
        obs = env.reset()
        model_client.reset(task_description)
        for step in range(max_steps):
            action = model_client.step(obs, step)  # 내부에서 청킹/앙상블 관리
            obs, reward, done, info = env.step(action)
            if done: break
        total_successes += info["success"]
```

**BEHAVIOR 예시** (`start_behavior_env.py`):
- OmniGibson의 `Evaluator` 클래스 사용
- `evaluator.step()` → (terminated, truncated) 반환
- `metric.start_callback(env)` / `metric.end_callback(env)` / `metric.gather_results()` 패턴
- 인스턴스별 JSON 결과 파일, 비디오 레코딩

**액션 청킹**: 클라이언트 사이드에서 관리.
```python
# model2libero_interface.py / model2behavior_interface.py
if self.current_step % action_chunk_size == 0:
    response = self.client.infer(vla_input)  # 서버 쿼리
    self.raw_actions = self.unnormalize_actions(response["data"]["normalized_actions"][0])
raw_action = self.raw_actions[self.current_step % action_chunk_size]
```

### 통신 구조

> **핵심 발견: starVLA는 우리 설계와 동일한 WebSocket + msgpack 프로토콜 사용.**

**서버** (`WebsocketPolicyServer`):
- asyncio 기반, `websockets` 라이브러리
- msgpack 바이너리 직렬화
- 압축 없음 (`compression=None`), 메시지 크기 무제한 (`max_size=None`)
- 유휴 타임아웃 지원 (`_idle_watchdog`)

**클라이언트** (`WebsocketClientPolicy`):
- 동기 websockets (`websockets.sync.client`)
- 연결 대기 최대 300초 (`_wait_for_server`)
- API Key 기반 인증 헤더 지원

**메시지 프로토콜:**
```
요청: {"type": "infer"|"ping", "request_id": "...", "payload": {...}}
응답: {"status": "ok"|"error", "ok": true|false, "type": "...", "request_id": "...", "data": {...}}
```

**msgpack_numpy.py** (58줄, 핵심):
```python
def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    ...

Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
```
- numpy 배열을 `{__ndarray__: True, data: raw_bytes, dtype: str, shape: tuple}`로 직렬화
- pickle 대비 ~4배 빠르다고 명시
- Void/Object/Complex dtype은 명시적으로 거부 (보안)

### 환경 격리

**Docker 격리 없음.** 벤치마크 환경이 평가 스크립트와 동일 프로세스에서 실행.

서버와 클라이언트는 별도 프로세스로 실행, WebSocket으로 연결:
- `run_policy_server.sh` — 모델 서버 실행
- `eval_libero.sh` / `start_behavior_env.sh` — 벤치마크 클라이언트 실행

### 설정/CLI

- **OmegaConf 기반 YAML 설정** — `config.yaml` per run
- CLI 오버라이드: `accelerate launch` + argparse
- **셸 스크립트가 주요 진입점** — 통합 CLI 도구 없음
- 설정을 체크포인트와 함께 저장하여 재현성 확보

### 결과 수집

**최소한의 벤치마크별 결과 수집:**
- LIBERO: 단순 카운터 + `logging.info` per episode
- BEHAVIOR: OmniGibson metrics 프레임워크 (`metric.start_callback` / `metric.end_callback` / `metric.gather_results`) + 인스턴스별 JSON 파일
- 비디오 레코딩: `imageio.mimwrite` (LIBERO) 또는 OmniGibson video writer (BEHAVIOR)
- **벤치마크 간 결과 통합/집계 없음**

## 프로젝트별 심층 분석

### 1. WebSocket + msgpack 통신 패턴 (검증 결과)

starVLA의 `msgpack_numpy.py`는 우리 설계에서 제안한 접근법과 본질적으로 동일. 프로덕션에서 검증된 58줄의 구현:

| 측면 | starVLA 구현 | 시사점 |
|------|-------------|--------|
| 직렬화 | `obj.tobytes()` + dtype.str + shape | 우리 설계의 numpy 직렬화 전략 검증 |
| 보안 | Void/Object/Complex dtype 거부 | pickle 대비 안전, 우리도 채택 필요 |
| 성능 | pickle 대비 ~4x 빠름 (문서 명시) | 고차원 action 공간에 적합 |
| 메시지 크기 | `max_size=None` (무제한) | 이미지 포함 시 필요 |
| 압축 | `compression=None` | LAN 환경에서는 압축 오버헤드가 더 클 수 있음 |

### 2. model2{bench}_interface 패턴 분석

각 인터페이스 파일은 `ModelClient` 클래스로, 공통 구조:

```python
class ModelClient:
    def __init__(self):
        self.client = WebsocketClientPolicy(host, port)   # WebSocket 연결
        self.action_norm_stats = self.get_action_stats()   # 정규화 통계 로드
        self.action_ensembler = AdaptiveEnsembler(...)     # 앙상블러 설정
    def reset(self, task_description): ...                  # 내부 상태 리셋
    def step(self, obs, step): ...                          # 청킹 경계에서만 서버 쿼리
    @staticmethod
    def unnormalize_actions(normalized, stats): ...         # 액션 비정규화
    @staticmethod
    def get_action_stats(key, ckpt_path): ...               # 통계 파일 로드
```

**벤치마크별 차이점:**
- LIBERO: 7-DoF (`world_vector[3]`, `rotation_delta[3]`, `open_gripper[1]`)
- BEHAVIOR: 23-DoF (`base[3]`, `torso[4]`, `left_arm[7]`, `left_gripper[1]`, `right_arm[7]`, `right_gripper[1]`)
- SimplerEnv: 별도의 `model2simpler_interface.py`

→ **시사점**: 이 "인터페이스" 레이어의 공통 부분을 추출하면 우리의 Benchmark ABC의 action 변환 로직이 됨.

### 3. Action Chunking + Adaptive Ensemble

**AdaptiveEnsembler** (44줄, `adaptive_ensemble.py`):
```python
class AdaptiveEnsembler:
    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        # 최신 예측과 이전 예측들 간 코사인 유사도 계산
        cos_similarity = dot_product / (norm_previous * norm_ref + 1e-7)
        # 지수 가중 평균
        weights = exp(alpha * cos_similarity)
        return weighted_average(weights, predictions)
```

- **Temporal ensemble과의 차이**: 고정 가중치가 아닌, 코사인 유사도 기반 적응적 가중치
- `alpha` 파라미터가 가중치 집중도 조절 (0이면 uniform, 높을수록 유사한 예측에 집중)
- `ChunkedAdaptiveEnsembler` 변형: multi-step chunk 대응

→ **해결됨**: 우리 `05_DESIGN_OUTLINE.md`의 `PredictModelServer.action_ensemble` 파라미터가 `"newest"` | `"average"` | `callable`을 지원한다. starVLA의 AdaptiveEnsembler(코사인 유사도 기반)는 `action_ensemble=callable`로 전달하면 된다. 즉, 고정 가중치 temporal ensemble(`"average"`)과 적응적 앙상블(callable)이 동일 인터페이스에서 공존한다.

### 4. Framework Registry

```python
# starVLA/model/tools.py
class Registry:
    def register(self, key):
        def decorator(cls):
            self._registry[key] = cls
            return cls
        return decorator

# starVLA/model/framework/__init__.py
for _, module_name, _ in pkgutil.iter_modules(pkg_path):
    importlib.import_module(f"{__name__}.{module_name}")
```

lm-eval-harness의 Registry와 비교:
| 측면 | lm-eval-harness | starVLA |
|------|----------------|---------|
| 등록 방식 | 데코레이터 | 데코레이터 |
| 자동 발견 | ConfigurableTask YAML | pkgutil.iter_modules |
| 에러 처리 | 경고 로그 | 경고 로그 (동일) |
| 스레드 안전성 | 고려됨 | 미고려 |
| 복잡도 | ~200줄 | ~25줄 |

### 5. 코드 중복 문제 (반면교사)

| 중복 코드 | 존재 파일 |
|-----------|----------|
| `unnormalize_actions()` | `base_framework.py`, `model2libero_interface.py`, `model2behavior_interface.py` |
| `_check_unnorm_key()` | `base_framework.py`, `model2libero_interface.py` |
| `get_action_stats()` | `model2libero_interface.py`, `model2behavior_interface.py` |
| WebSocket 연결 설정 | 모든 model2*_interface.py |
| 액션 앙상블 로직 | `adaptive_ensemble.py` 2개 별도 복사본 |

→ 우리 프레임워크의 핵심 가치: 이러한 중복을 Benchmark ABC + ModelServer 추상화로 제거.

### 6. 벤치마크 커버리지

| 벤치마크 | 디렉토리 | 특이사항 |
|----------|---------|---------|
| LIBERO | `examples/LIBERO/` | 가장 완성된 구현 |
| SimplerEnv | `examples/SimplerEnv/` | 자동 평가 스크립트 포함 |
| BEHAVIOR-1K | `examples/Behavior/` | OmniGibson Evaluator 활용 |
| RoboCasa | `examples/Robocasa_tabletop/` | 평가 + 학습 |
| RoboTwin | `examples/Robotwin/` | 평가 + 학습 |
| Calvin | `examples/calvin/` | 평가 + 학습 |

6개 벤치마크 모두 동일한 클라이언트-서버 패턴이나, **완전히 별도의 평가 스크립트**. 공유 평가 로직 없음.

## 코드 품질/성숙도

| 측면 | 평가 |
|------|------|
| 타입 힌트 | 보통 — 일부 파일에 존재, 다수 파일에 부재. Pydantic 모델 없음 |
| 테스트 | **없음** — 테스트 디렉토리 미존재 |
| CI | 미확인 |
| 문서화 | README 포괄적, eval_protocol.md 유용하나 간결 |
| 코드 품질 | 활발한 개발 중. 일부 거친 부분 (중복 메서드, 주석 처리된 디버그 코드, 비일관적 네이밍) |
| 성숙도 | 초중기 단계, 커뮤니티 주도 오픈소스 |

## 시사점

### 채택할 패턴

1. **WebSocket + msgpack 바이너리 직렬화** → 우리 **Protocol** 컴포넌트. starVLA의 프로덕션 사용으로 우리 설계 선택 검증. `msgpack_numpy.py` 접근법 직접 채택/수정 가능 (58줄, MIT 라이선스) → 우리 Protocol 직렬화 레퍼런스.
2. **numpy 배열 직렬화 프로토콜** — `{__ndarray__: True, data: bytes, dtype: str, shape: tuple}` 형식. pickle보다 안전하고 빠름 → 우리 Protocol의 msgpack 직렬화 구현 참고.
3. **Framework Registry 패턴** — 데코레이터 기반 등록 + auto-import → 우리의 Registry 패턴에 적용.
4. **`predict_action(examples) → {"normalized_actions": ndarray}` 인터페이스 계약** → 우리 `PredictModelServer.predict()` 인터페이스의 참고. 명확하고 일관된 모델 추론 인터페이스.
5. **액션 청킹 패턴** — starVLA는 클라이언트 사이드에서 `step % chunk_size == 0`일 때만 서버 쿼리하는 방식. 우리 설계에서는 이를 **서버 사이드**(`PredictModelServer.chunk_size`)로 이동하여 관심사를 더 깔끔하게 분리한다. 벤치마크(클라이언트)는 매 step마다 단일 action만 받으며, 청킹 로직을 알 필요가 없다.
6. **AdaptiveEnsembler 코사인 유사도 기반 앙상블** → 우리 `PredictModelServer.action_ensemble=callable` 옵션에 매핑. AdaptiveEnsembler 로직을 callable 함수로 전달하면 된다. 고정 가중치 temporal ensemble(`"average"`)의 개선된 대안으로 제공 가능.
7. **`from_pretrained(ckpt_path)`로 config + norm_stats + 가중치 일괄 로드** — `PredictModelServer.predict()` 내부의 모델 로딩 디테일로 참고. 재현성 보장.

### 회피할 패턴

1. **벤치마크별 ad-hoc 평가 스크립트 + 대규모 코드 중복** — `model2{bench}_interface` 코드 중복은 정확히 우리의 **Benchmark ABC** + **Connection**이 해결하는 문제. Benchmark ABC가 환경 인터페이스를 표준화하고, Connection이 통신을 프레임워크 수준에서 제공한다.
2. **공통 Benchmark ABC 부재** — 벤치마크 간 비교가 거의 불가능 → 우리의 Benchmark ABC(`get_tasks()`, `reset()`, `step()`, `make_obs()`, `is_done()`, `get_result()`)가 이를 해결.
3. **클라이언트 사이드 액션 비정규화** — 정규화/비정규화는 모델 서버 책임이어야 함 (관심사 분리).
4. **셸 스크립트로 서버+클라이언트 조율** → 우리의 **Orchestrator**가 Docker 기동, 서버 연결, runner 선택을 통합 관리.
5. **Docker 격리 없음** → 우리의 **벤치마크별 독립 Docker 이미지** 전략으로 환경 격리 및 재현성 확보.
6. **구조화된 결과 수집 없음** — `logging.info`로는 불충분 → 우리의 **Result Collector**가 에피소드/태스크/벤치마크 단위 메트릭 수집 및 통합 리포팅 제공.

### 우리 설계 컴포넌트와의 매핑 요약

| starVLA 요소 | 우리 설계 컴포넌트 | 비고 |
|-------------|-------------------|------|
| WebSocket + msgpack 통신 | **Protocol** | 동일 접근, 검증됨 |
| `msgpack_numpy.py` | **Protocol** 직렬화 레퍼런스 | 58줄, MIT, 직접 채택/수정 가능 |
| `FRAMEWORK_REGISTRY` | **Registry** 패턴 | 데코레이터 기반 등록 |
| `predict_action()` 인터페이스 | `PredictModelServer.predict()` | 유사한 계약 |
| 클라이언트 사이드 액션 청킹 | `PredictModelServer.chunk_size` (서버 사이드) | 우리는 서버에서 관리 |
| `AdaptiveEnsembler` | `PredictModelServer.action_ensemble=callable` | callable로 전달 |
| `from_pretrained()` | `predict()` 내부 모델 로딩 디테일 | — |
| `model2{bench}_interface` 코드 중복 | **Benchmark ABC** + **Connection** | 우리가 해결하는 핵심 문제 |
| Docker 격리 없음 | **벤치마크별 Docker 이미지** | 우리의 격리 전략 |
| 결과 통합 없음 | **Result Collector** | 에피소드/태스크/벤치마크 메트릭 |

### 열린 질문

1. starVLA의 `msgpack_numpy.py`를 우리 프로젝트에 직접 채택/수정할 것인가? (MIT 라이선스이므로 가능)
2. `model2{bench}_interface` 패턴의 공통 부분을 어떻게 추출하여 우리의 Benchmark ABC에 반영할 것인가?
3. ~~AdaptiveEnsembler의 코사인 유사도 방식과 우리 temporal ensemble averaging 방식의 관계는?~~ → **해결됨**: `PredictModelServer.action_ensemble`이 `"newest"` | `"average"` | `callable`을 지원. AdaptiveEnsembler는 `callable`로 전달하면 된다. 두 방식이 동일 인터페이스에서 공존.
4. ~~`action_chunk_size`를 model config에서 읽는 starVLA 방식 vs 우리 설계의 ModelServer가 관리하는 방식 중 어느 것이 적합한가?~~ → **해결됨**: `PredictModelServer.chunk_size` 파라미터로 서버가 관리. 벤치마크(클라이언트)는 청킹을 알 필요 없음.
5. ~~`predict_action`이 정규화된 액션을 반환하는 starVLA 방식 vs 비정규화된 액션을 반환하는 방식 중 어느 것이 적합한가?~~ → **참고**: 우리 설계는 `dict[str, Any]` 페이로드를 사용하므로 정규화/비정규화 여부를 프레임워크가 강제하지 않는다. 모델 서버가 자체적으로 선택할 수 있다.

