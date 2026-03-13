# X-VLA — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/2toINF/X-VLA |
| Latest Commit | `6bc2513` / 2026-01-28 |
| 규모 | Python 59개 파일, ~15,315 라인 |
| 라이선스 | Apache License 2.0 |
| ArXiv | 2510.10274 (ICLR 2026) |
| 주요 목적 | Cross-embodiment VLA — Florence2 인코더 + SoftPromptedTransformer + Flow-Matching 액션 헤드. 소프트 프롬프트와 도메인별 파라미터로 다중 체현(로봇) 동시 학습/추론 |
| 우리 프로젝트에서의 참고 포인트 | 01_PROPOSAL.md: "VLA 모델의 벤치마크 연동 코드 참고" |

## 프로젝트 구조

```
X-VLA/
├── models/
│   ├── modeling_xvla.py           # XVLA(PreTrainedModel) — 핵심 모델 + FastAPI /act 엔드포인트
│   ├── configuration_xvla.py      # XVLAConfig(PretrainedConfig)
│   ├── processing_xvla.py         # XVLAProcessor(ProcessorMixin) — 멀티뷰 이미지+언어 토크나이저
│   ├── transformer.py             # SoftPromptedTransformer — DomainAwareLinear, TransformerBlock
│   └── action_hub.py              # ACTION_REGISTRY + BaseActionSpace ABC + 4개 구현체
├── datasets/
│   ├── dataset.py                 # InfiniteDataReader(IterableDataset) — 가중 다중도메인 샘플링
│   ├── domain_config.py           # DATA_WEIGHTS, DATA_DOMAIN_ID (~30개 도메인 매핑)
│   └── domain_handler/
│       ├── registry.py            # _REGISTRY dict — 핸들러 클래스 매핑 (~30 항목)
│       ├── base.py                # DomainHandler ABC + BaseHDF5Handler (보간 기반 샘플링)
│       ├── lerobot_agibot.py      # AGIBot LeRobot 형식 핸들러
│       ├── simulations.py         # LiberoHandler, CalvinHandler, VLABenchHandler 등
│       └── ...                    # 8개 추가 핸들러 모듈
├── evaluation/
│   ├── libero/libero_client.py    # LIBERO 평가 클라이언트 (457줄, 가장 상세)
│   ├── calvin/calvin_client.py    # CALVIN ABC→D 평가 클라이언트 (234줄)
│   ├── vlabench/vlabench_client.py # VLABench 평가 클라이언트 (228줄)
│   ├── robotwin-2.0/client.py     # RoboTwin 양팔 평가 클라이언트 (413줄)
│   └── simpler/                   # SimplerEnv — 태스크별 개별 스크립트 (12개 파일)
│       ├── google-VA/             # Google Robot Visual Appearance (4 클라이언트)
│       ├── google-VM/             # Google Robot Visual Matching (4 클라이언트)
│       └── WidowX/               # WidowX Bridge (4 클라이언트)
├── train.py                       # Accelerate 기반 학습 (282줄) — 4-group 옵티마이저
├── peft_train.py                  # LoRA 미세조정 (297줄) — r=8, alpha=16
└── deploy.py                      # FastAPI 서버 런처 + SLURM 지원 (164줄)
```

## 핵심 분석

### 벤치마크 추상화

**공통 Benchmark ABC 없음.** 각 벤치마크는 `evaluation/` 내 완전히 독립된 클라이언트 스크립트로 존재.

| 벤치마크 | 파일 | domain_id | 액션 차원 | 특이사항 |
|----------|------|-----------|-----------|----------|
| LIBERO | `libero_client.py` | 3 | 10 (pos3+rot6d+grip1) | 6D→axis-angle 변환, `LiberoAbsActionProcessor` |
| CALVIN | `calvin_client.py` | 2 | 10 (pos3+rot6d+grip1) | `CalvinBaseModel` 상속, euler→6D→quat 변환 |
| VLABench | `vlabench_client.py` | 8 | 10 (pos3+rot6d+grip1) | 3-view (main+front+wrist), 좌표계 오프셋 보정 |
| RoboTwin | `robotwin-2.0/client.py` | 6 | 20 (양팔 각 10) | 50개 태스크, expert_check 사전 검증 |
| SimplerEnv | `simpler/*.py` | 1/4/5 | 다양 | **태스크당 별도 파일** — 12개 스크립트, 코드 중복 심각 |

**공통 패턴**: 모든 클라이언트가 동일한 HTTP POST 기반 통신을 사용하지만, 이를 추상화하는 공통 베이스 클래스는 없음. `action_plan` 큐잉, `proprio` 관리, 회전 변환 로직이 파일마다 복사-붙여넣기됨.

### 모델 통합 패턴

**HuggingFace-native**: `XVLA(PreTrainedModel)` + `XVLAProcessor(ProcessorMixin)` + `XVLAConfig(PretrainedConfig)`.

```python
# models/modeling_xvla.py
class XVLA(PreTrainedModel):
    config_class = XVLAConfig
    
    def __init__(self, config: XVLAConfig):
        self.vlm = Florence2ForConditionalGeneration(...)  # 인코더만 사용
        self.transformer = SoftPromptedTransformer(config)
        self.action_space = build_action_space(config.action_space_type)
    
    def forward(self, batch):  # 학습: flow-matching loss
        noise = torch.randn_like(gt)
        t = torch.rand(B, 1, 1)
        x_t = t * noise + (1 - t) * gt       # ← flow-matching interpolation
        pred = self.transformer(x_t, vlm_out, ...)
        loss = self.action_space.compute_loss(pred, gt, ...)
    
    def generate_actions(self, ...):  # 추론: iterative denoising
        x_t = torch.randn(1, num_actions, action_dim)
        for i in range(steps):  # 기본 10 steps
            t = 1 - i / steps
            pred = self.transformer(x_t, vlm_out, ...)
            x_t = x_t + (1/steps) * (gt_pred - x_t)  # Euler step
```

**서빙 내장**: XVLA 클래스 자체에 FastAPI 앱이 내장 (`_build_app()` 메서드):

```python
# models/modeling_xvla.py — XVLA._build_app()
app = FastAPI()
@app.post("/act")
async def act(request: Request):
    data = await request.json()
    images = [json_numpy.loads(data[f"image{i}"]) for i in range(3)]
    proprio = json_numpy.loads(data["proprio"])
    domain_id = data["domain_id"]
    actions = model.generate_actions(images, proprio, domain_id, ...)
    return {"action": actions.tolist()}
```

이 설계의 장단점:
- **장점**: `from_pretrained()` → `model._build_app()` → `uvicorn.run()` 3줄로 배포 완료
- **단점**: 모델과 서빙 코드의 강결합. 다른 서빙 방식(gRPC, WebSocket 등) 지원이 어려움

### 에피소드 실행 루프

각 클라이언트의 에피소드 루프 패턴 (LIBERO 기준):

```
for task in suite:
    for episode in range(num_episodes):
        env, lang, obs = init_env(task, episode)
        policy.reset()
        for step in range(horizon):
            action = policy.step(obs, lang)  # HTTP POST → 서버 → 액션 청크
            obs, reward, done, info = env.step(action)
```

**액션 청킹**: 서버에서 T개 액션을 한번에 반환 → 클라이언트가 deque에 저장 → 한 스텝씩 pop하여 실행. 큐가 빌 때만 서버에 재요청.

> **우리 설계와의 대응**: 이 전체 메커니즘(청크 버퍼 관리, 한 스텝씩 pop, 재요청 타이밍)은 우리 `PredictModelServer`의 내장 `chunk_size` 파라미터가 자동 처리한다. 프레임워크가 chunk buffer를 관리하고 Connection(클라이언트)에는 매 step 단일 action만 전달하므로, 벤치마크/클라이언트 코드에서 청크 관리 로직이 완전히 제거된다.

### 통신 구조

**HTTP POST + json_numpy**:
- 프로토콜: REST (POST `/act`)
- 직렬화: `json_numpy` (NumPy 배열 → JSON 호환 형식)
- 페이로드: `{image0, image1, [image2], proprio, language_instruction, domain_id, steps}`
- 응답: `{"action": [[...], [...], ...]}`  (T × action_dim 2D 리스트)

starVLA의 WebSocket + msgpack과 비교:
| | X-VLA | starVLA |
|---|-------|---------|
| 프로토콜 | HTTP POST (동기) | WebSocket (비동기) |
| 직렬화 | json_numpy | msgpack + numpy |
| 연결 | stateless (매 요청 새 연결) | persistent connection |
| 지연 | 높음 (HTTP 오버헤드) | 낮음 |

### 환경 격리

**격리 없음.** 평가 클라이언트가 벤치마크 환경을 직접 import하여 같은 프로세스에서 실행.

```python
# evaluation/libero/libero_client.py
from libero.libero.envs import OffScreenRenderEnv  # 환경 직접 import
env = OffScreenRenderEnv(**env_args)               # 같은 프로세스에서 생성
```

Docker, subprocess 격리 모두 없음. 환경 의존성 충돌 시 수동으로 해결해야 함.

### 설정 / CLI

**argparse 기반, 벤치마크별 개별 CLI**:

| 파일 | 주요 인자 |
|------|-----------|
| `train.py` | `--data_path`, `--data_weights`, `--model_path`, `--lr`, `--warmup_steps`, ... |
| `deploy.py` | `--model_path`, `--num_gpus`, `--port` |
| `libero_client.py` | `--connection_info\|--server_ip+port`, `--task_suites`, `--eval_time` |
| `calvin_client.py` | `--connection_info\|--server_ip+port`, `--eval_start/end` |

서버-클라이언트 연결 방식: `deploy.py`가 `info.json` 파일 생성 → 클라이언트가 `--connection_info`로 읽음 (SLURM 환경 대응).

### 결과 수집

**JSON Lines append 방식**:

```python
# evaluation/libero/libero_client.py
with open(save_name, 'a+') as f:
    f.write(json.dumps(metrics) + "\n")  # {"sim/libero_goal/task_name": 1.0}
```

- 에피소드별 `results.json`에 JSON Lines append
- 최종 summary도 같은 파일에 `json.dump(result_dict)` append
- 비디오: 에피소드별 `.mp4` 저장 (`imageio.mimsave`)
- 통합 결과 집계 도구 없음 — 각 클라이언트가 자체 로깅

## 프로젝트별 심층 분석

### 1. Flow-Matching 액션 생성

X-VLA는 autoregressive 토큰 생성이 아닌 **flow-matching** (연속 정규화 흐름)으로 액션을 생성:

**학습**: ground-truth 액션과 가우시안 노이즈 사이의 선형 보간 → velocity 예측
```python
# modeling_xvla.py — forward()
noise = torch.randn_like(gt)
t = torch.rand(B, 1, 1)
x_t = t * noise + (1 - t) * gt           # noisy interpolant
velocity_pred = self.transformer(x_t, t, vlm_features, ...)
loss = MSE(velocity_pred, noise - gt)     # flow-matching objective
```

**추론**: 순수 노이즈에서 시작 → N번 Euler step으로 디노이징
```python
# modeling_xvla.py — generate_actions()
x_t = torch.randn(1, num_actions, action_dim)  # 순수 노이즈
for i in range(steps):                           # 기본 10 steps
    t = 1 - i / steps
    pred = self.transformer(x_t, t, vlm_features, ...)
    x_t = x_t + (1/steps) * pred                 # Euler integration
```

**비교**: Pi0/Pi0.5(OpenPI)도 flow-matching이지만 VLM 디코더 내부에 통합. X-VLA는 Florence2 인코더만 사용하고 별도 transformer가 flow-matching 수행 → 더 작은 모델(0.9B vs Pi0의 3B).

### 2. SoftPromptedTransformer — 도메인 조건화 메커니즘

cross-embodiment의 핵심: **도메인별 학습 가능 파라미터**로 공유 백본을 조건화.

**3가지 도메인 조건화 경로**:

**(a) Soft Prompt Hub**: 도메인별 학습 가능 시퀀스 토큰
```python
# transformer.py
self.soft_prompt_hub = nn.ParameterDict({
    domain_name: nn.Parameter(torch.randn(len_soft_prompts, hidden_size))
    for domain_name in domain_names
})
# 추론 시: soft_prompts를 input sequence에 concatenate
sequence = [action_tokens, vlm_features, aux_visual, soft_prompts]
```

**(b) DomainAwareLinear**: 도메인별 가중치/편향을 `nn.Embedding`으로 저장
```python
# transformer.py
class DomainAwareLinear(nn.Module):
    def __init__(self, in_features, out_features, num_domains):
        self.weight = nn.Embedding(num_domains, out_features * in_features)
        self.bias = nn.Embedding(num_domains, out_features)

    def forward(self, x, domain_id):
        W = self.weight(domain_id).view(out_features, in_features)
        b = self.bias(domain_id).view(out_features)
        return F.linear(x, W, b)
```
→ 액션 인코더(action→hidden)와 액션 디코더(hidden→action)에 사용. 도메인마다 다른 액션 공간을 하나의 모델로 처리.

**(c) Timestep Embedding**: flow-matching의 시간 `t`를 sinusoidal embedding으로 변환 → AdaLN (adaptive layer norm)에 주입.

### 3. ACTION_REGISTRY — 액션 공간 추상화

```python
# action_hub.py
ACTION_REGISTRY: Dict[str, Type[BaseActionSpace]] = {}

def register_action(name: str):
    def decorator(cls):
        ACTION_REGISTRY[name] = cls
        return cls
    return decorator

class BaseActionSpace(ABC):
    @abstractmethod
    def preprocess(self, raw_action: Tensor) -> Tensor: ...
    @abstractmethod
    def postprocess(self, pred_action: Tensor) -> Tensor: ...
    @abstractmethod
    def compute_loss(self, pred, gt, ...) -> Tensor: ...
```

| 이름 | 차원 | 구성 | 사용처 |
|------|------|------|--------|
| `ee6d` | 20 (입력 10×2) | pos3 + rot6d + grip1 (×2, current+past) | LIBERO, CALVIN, VLABench |
| `joint` | 14 (7×2) | 7-DoF joint (×2) | 관절 제어 로봇 |
| `agibot_ee6d` | 20 | AGIBot 전용 6D EE 포맷 | AGIBot 로봇 |
| `auto` | 가변 | 도메인 config에서 차원 자동 결정 | 범용 |

**loss 분리**: `compute_loss()`에서 위치/회전/그리퍼 각각에 별도 가중치 적용 (`loss_pos`, `loss_rot`, `loss_grip`). 이는 6D 회전의 스케일이 위치/그리퍼와 다르기 때문.

### 4. 데이터셋 로딩 — DomainHandler 레지스트리

**2중 레지스트리 구조**:

1. **`domain_config.py`**: 도메인 이름 → 가중치/ID 매핑 (학습 시 샘플링 비율 결정)
   ```python
   DATA_WEIGHTS = {"libero": 1.0, "Calvin": 1.5, "Bridge": 2.0, ...}
   DATA_DOMAIN_ID = {"libero": 3, "Calvin": 2, "VLABench": 8, ...}
   ```

2. **`domain_handler/registry.py`**: 도메인 이름 → Handler 클래스 매핑 (데이터 로딩 방식 결정)
   ```python
   _REGISTRY = {"libero": LiberoHandler, "Calvin": CalvinHandler, ...}
   ```

**BaseHDF5Handler의 보간 기반 샘플링**:
- HDF5에서 좌/우 궤적(left_traj, right_traj)과 타임스탬프 로딩
- `scipy.interp1d`로 시간축 보간 → 원하는 시점의 액션을 연속적으로 쿼리
- 정적(static) 구간 자동 스킵: 연속 프레임 간 변화 < 1e-5이면 건너뜀
- 언어 증강: `lang_aug_map`으로 instruction 다양화

**InfiniteDataReader**:
```python
# dataset.py
class InfiniteDataReader(IterableDataset):
    def __iter__(self):
        while True:
            domain = weighted_random_choice(self.domains, self.weights)
            handler = get_handler_cls(domain.name)(domain.meta, num_views)
            traj_idx = random.randint(0, domain.num_trajs - 1)
            yield from handler.iter_episode(traj_idx, ...)
```
무한 반복 + 가중 도메인 샘플링 → Accelerate의 DataLoader에 직접 연결.

### 5. 학습 파이프라인

**4-group 옵티마이저** (각 그룹에 독립적 LR):
```python
# train.py
param_groups = [
    {"params": vlm_params,              "lr": 1e-5},   # Florence2 backbone
    {"params": transformer_core_params, "lr": 5e-5},   # Transformer blocks
    {"params": soft_prompt_params,      "lr": 1e-4},   # Soft prompts
    {"params": action_head_params,      "lr": 1e-4},   # Action encoder/decoder
]
```

**Freeze-Warmup 스케줄**: 처음 N 스텝 동안 VLM+Transformer 동결, soft_prompts+action_heads만 학습 → 이후 전체 unfreeze + cosine LR decay.

**LoRA 미세조정** (`peft_train.py`): 풀 학습 대신 LoRA(r=8, alpha=16)를 all-linear 모듈에 적용. `soft_prompt_hub`, `action_encoder`, `action_decoder`는 `modules_to_save`로 지정하여 LoRA와 별도로 풀 학습.

### 6. 서버-클라이언트 배포 아키텍처

```
┌─────────────────────┐          HTTP POST /act          ┌────────────────────┐
│   deploy.py         │ ◀────────────────────────────────│ evaluation/         │
│   (FastAPI Server)  │          json_numpy              │ *_client.py        │
│                     │ ────────────────────────────────▶ │ (벤치마크별 클라이언트)│
│  model.generate()   │          {"action": [...]}       │ env.step(action)   │
│  info.json 생성      │                                  │                    │
└─────────────────────┘                                  └────────────────────┘
```

**`deploy.py`의 SLURM 지원**:
```python
# deploy.py
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
with open("info.json", "w") as f:
    json.dump({"host": ip, "port": port}, f)
# → 클라이언트가 info.json을 polling하여 서버 주소 획득
```

SLURM에서 서버/클라이언트를 다른 노드에 배치 가능. 클라이언트는 `info.json` 파일이 생길 때까지 spinner로 대기.

## 코드 품질 / 성숙도

| 항목 | 평가 |
|------|------|
| **타입 힌트** | 중간. LIBERO 클라이언트는 상세 (`typing` 적극 사용), 나머지 파일은 부분적 |
| **테스트** | 없음. CI/CD 없음, 테스트 파일 없음 |
| **문서화** | README.md (설치/학습/배포 가이드), 코드 내 docstring은 핵심 파일만 |
| **에러 처리** | HTTP 클라이언트에 `raise_for_status()` 있으나, 일부 파일에서 try/except가 주석 처리됨 |
| **코드 중복** | 심각. 회전 변환(quat↔6D↔euler↔axis-angle), HTTP 클라이언트 로직, CLI 파싱이 5개 평가 스크립트에 복사됨 |
| **하드코딩** | RoboTwin 클라이언트에 절대 경로(`/home/dodo/fyc/RoboTwin`) 하드코딩 |
| **HuggingFace 호환** | 우수. `from_pretrained()`, `save_pretrained()`, `AutoModel.register()` 완전 지원 |

## 시사점

### 채택할 패턴

1. **Flow-matching 서빙 인터페이스 → `PredictModelServer.predict()`**: `generate_actions()`의 `(images, proprio, domain_id, steps)` → `action_chunk` 인터페이스가 깔끔함. 우리 설계에서는 `PredictModelServer.predict(obs) → {"actions": ndarray}`가 이에 대응. Flow-matching의 denoising step 수, domain_id 등 모델 내부 파라미터는 `predict()` 구현 내부의 디테일이며 프레임워크가 관여하지 않음.

2. **ACTION_REGISTRY + BaseActionSpace → Benchmark ABC의 `step()` 책임**: 액션 공간별 `preprocess`/`postprocess` 분리가 우수하나, 우리 프레임워크에서는 별도의 `ActionPostProcessor`나 `ActionRegistry`를 두지 않음. 액션 공간 변환(6D→axis-angle, joint→EE 등)은 각 Benchmark의 `step()` 내부에서 처리 — 벤치마크별 액션 전처리/후처리는 Benchmark 구현자의 책임.

3. **DomainHandler 패턴 → Benchmark ABC의 `make_obs()`**: 데이터셋 형식 다양성을 레지스트리 + ABC로 통일한 패턴. 우리 설계에서는 `Benchmark.make_obs(raw_obs, task) → dict[str, Any]`가 벤치마크별 observation 형식 통일을 담당. 각 Benchmark 구현체가 환경 고유의 raw observation을 모델 서버에 전송할 dict로 변환.

4. **info.json 기반 서버 디스커버리 → YAML config `server.url`**: SLURM/클러스터 환경에서의 서버-클라이언트 연결 방식으로 실용적이나, 우리 설계에서는 YAML 설정 파일의 `server.url: "ws://..."` 필드로 더 단순하게 해결. 동적 디스커버리가 필요하면 YAML에 info.json 경로를 지정하는 방식으로 확장 가능.

5. **domain_id 기반 조건부 실행 → 모델-불가지론적 설계**: 우리 프레임워크는 모델-불가지론적이므로 domain_id 디스패치를 프레임워크 레벨에서 지원하지 않음. 특정 모델이 domain_id를 필요로 하면, Benchmark가 `make_obs()`에서 observation dict에 포함시키면 됨 (예: `{"domain_id": 3, "agentview_image": ...}`). 모델 서버의 `predict()` 구현이 이를 읽어 사용.

### 회피할 패턴

1. **서빙 코드의 모델 내장 → ModelServer ABC로 분리**: `XVLA` 클래스에 FastAPI 앱이 내장된 것은 단일 책임 원칙 위반. 우리 설계에서는 `ModelServer`가 별도 ABC로 존재하며, 모델 로딩/추론(`predict()`)과 서빙(WebSocket + msgpack Protocol)이 완전히 분리됨. 연구자는 `predict()`만 구현하고, 서빙 인프라는 프레임워크가 제공.

2. **평가 클라이언트 간 코드 중복 → Benchmark ABC + Connection**: 5개 벤치마크 클라이언트에 HTTP 통신, 회전 변환, 액션 큐잉 로직이 복사됨. 우리 설계에서는 통신은 `Connection`이, 에피소드 루프는 `EpisodeRunner`가, 환경 인터페이스는 `Benchmark` ABC가 각각 담당하여 코드 중복을 구조적으로 제거.

3. **domain_id 하드코딩 → YAML config**: 클라이언트마다 `"domain_id": 3`, `"domain_id": 2` 등 매직 넘버. 우리 설계에서는 YAML 설정 파일에서 벤치마크별 파라미터를 관리하며, 필요 시 Benchmark 구현체가 설정에서 읽어 observation dict에 포함.

4. **절대 경로 하드코딩**: RoboTwin 클라이언트에 `/home/dodo/fyc/RoboTwin` 등 개발자 환경 경로가 그대로 노출. 우리 설계에서는 Docker 격리 + YAML config로 환경 경로를 관리.

5. **SimplerEnv의 태스크별 파일 폭발 → 단일 Benchmark 클래스**: 12개의 거의 동일한 클라이언트 스크립트. 우리 설계에서는 벤치마크당 하나의 `Benchmark` 구현 클래스가 `get_tasks()`로 태스크 목록을 반환하고, 태스크별 차이는 task dict의 파라미터로 처리.

### 열린 질문

1. **Flow-matching 모델의 추론 step 수와 품질 트레이드오프**: 기본 10 steps인데, step 수를 줄이면 (예: 1-2 steps) 실시간 제어가 가능한가? → **우리 설계에서는 모델 구현 내부의 디테일.** `PredictModelServer.predict()` 내부에서 연구자가 step 수를 자유롭게 설정하며, 프레임워크는 이를 노출하거나 관여하지 않음. 추론 속도가 느려지면 `AsyncEpisodeRunner`의 hold_policy가 자동으로 대응.

2. **domain_id vs model configuration**: → **우리 설계로 해결됨.** 프레임워크는 모델-불가지론적이므로 domain_id를 프레임워크 레벨에서 다루지 않음. 모델이 domain_id를 필요로 하면, Benchmark가 `make_obs()`에서 observation dict에 포함 (예: `{"domain_id": 3, ...}`). `dict[str, Any]` 페이로드이므로 어떤 모델 특화 파라미터든 자유롭게 전달 가능.

3. **액션 청크 크기의 벤치마크별 최적화**: X-VLA는 서버가 고정 크기 청크를 반환하고 클라이언트가 순차 실행. 청크 크기가 벤치마크 물리 시뮬레이션 주기와 맞지 않으면 어떤 문제가 생기는지 — 이는 `PredictModelServer`의 `chunk_size`와 `action_ensemble` 파라미터로 모델 서버 측에서 조정하며, Benchmark/Connection은 항상 단일 action만 수신.

4. **multi-view 이미지 처리의 표준화**: X-VLA는 image0/1/2로 고정 인덱싱하는데, 벤치마크마다 카메라 뷰 이름이 다름 (agentview, rgb_static, head_camera, ...). → **우리 설계에서는 `dict[str, Any]`로 해결.** `Benchmark.make_obs()`가 환경 고유의 카메라 뷰 키를 모델이 기대하는 키로 매핑. 프레임워크가 키 이름을 강제하지 않으므로 벤치마크 구현자가 자유롭게 결정.

