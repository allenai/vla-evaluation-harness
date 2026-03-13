# InternVLA-M1 — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/OpenGVLab/InternVLA-M1 |
| Latest Commit | `21e6e8f` / 2026-02-11 |
| 규모 | Python 68개 파일, ~15,087 라인 |
| 라이선스 | MIT License |
| ArXiv | 2510.13778 |
| 주요 목적 | Dual-Encoder VLA — Qwen2.5-VL + DINOv2 + LayerwiseQFormer + DiT Gaussian Diffusion 액션 헤드. VLA 학습/추론 + VLM 공동학습 지원 |
| 원류 | starVLA의 **upstream** 프로젝트 (starVLA가 이 프로젝트에서 fork) |
| 우리 프로젝트에서의 참고 포인트 | 01_PROPOSAL.md: "위와 같은 축의 다른 구현 비교" (X-VLA와 동일 축) |

## 프로젝트 구조

```
InternVLA-M1/
├── InternVLA/                        # 핵심 패키지
│   ├── model/
│   │   ├── framework/
│   │   │   ├── M1.py                 # InternVLA_M1(baseframework) — 4-컴포넌트 메인 모델
│   │   │   ├── base_framework.py     # baseframework(nn.Module) ABC — from_pretrained, 정규화
│   │   │   ├── __init__.py           # build_framework() — if-based 디스패치 (레지스트리 없음)
│   │   │   └── share_tools.py        # OmegaConf 기반 설정 유틸리티
│   │   ├── modules/
│   │   │   ├── action_model/
│   │   │   │   └── DiTActionHeader.py  # ActionModel — DiT + GaussianDiffusion + DDIM
│   │   │   ├── vlm/
│   │   │   │   └── QWen2_5.py        # _QWen_VL_Interface — Qwen2.5-VL 래퍼 (Flash Attention 2)
│   │   │   ├── projector/
│   │   │   │   └── QFormer.py        # LayerwiseQFormer — 다층 교차 어텐션 프로젝터
│   │   │   └── dino_model/
│   │   │       └── dino.py           # DINOv2BackBone — 멀티뷰 병렬 전처리
│   │   └── tools.py                  # 학습 가능 모듈 자동 탐색
│   ├── dataloader/
│   │   ├── lerobot_datasets.py       # LeRobot 기반 VLA 데이터셋 (NAMED_MIXTURES)
│   │   └── vlm_datasets.py           # LLaVA JSON 형식 VLM 데이터셋 (651줄)
│   ├── training/
│   │   ├── train_internvla.py        # VLATrainer — 단일 VLA 학습
│   │   ├── train_internvla_cotrain.py # VLAMTrainer — VLA+VLM 공동학습
│   │   └── trainer_utils/
│   │       └── metrics.py            # TrainerUtils, build_param_lr_groups, PCGrad
│   └── config/training/
│       └── internvla_cotrain_oxe.yaml # 대표 학습 설정 (101줄)
├── deployment/
│   └── model_server/
│       ├── server_policy_M1.py       # WebSocket 서버 진입점 (66줄)
│       └── tools/
│           ├── websocket_policy_server.py  # WebsocketPolicyServer (asyncio)
│           ├── websocket_policy_client.py  # WebsocketClientPolicy (sync)
│           ├── msgpack_numpy.py            # msgpack + numpy 직렬화
│           └── image_tools.py              # 이미지 변환 유틸 (resize_with_pad, to_pil_preserve)
├── examples/                         # 벤치마크별 평가 코드
│   ├── LIBERO/
│   │   ├── eval_libero.py            # LIBERO 평가 루프 (298줄, tyro CLI)
│   │   └── model2libero_interface.py # M1Inference 클라이언트 (223줄)
│   └── SimplerEnv/
│       ├── model2simpler_interface.py # M1Inference 클라이언트 (270줄)
│       └── adaptive_ensemble.py       # AdaptiveEnsembler
└── scripts/                          # SLURM/shell 실행 스크립트
```

## 핵심 분석

### 벤치마크 추상화

**공통 Benchmark ABC 없음.** X-VLA와 동일한 패턴: 벤치마크별 독립 스크립트.

| 벤치마크 | 파일 | 액션 차원 | 특이사항 |
|----------|------|-----------|----------|
| LIBERO | `eval_libero.py` + `model2libero_interface.py` | 7 (pos3+rot3+grip1) | 이미지 180° 회전, action chunking |
| SimplerEnv | `model2simpler_interface.py` | 7 (pos3+rot3+grip1) | Euler→axis-angle 변환, 로봇별 sticky gripper 로직 |

**공통 패턴**: 모든 클라이언트가 `M1Inference` 클래스를 구현하되, **벤치마크별 별도 파일**로 존재. `reset()`, `step()` 인터페이스는 유사하나 추상 베이스 클래스 없음. X-VLA (5개 벤치마크)에 비해 통합 수가 적어 중복이 상대적으로 관리 가능한 수준.

### 모델 통합 패턴

**4-컴포넌트 Dual-Encoder 아키텍처**:

```python
# InternVLA/model/framework/M1.py
class InternVLA_M1(baseframework):
    def __init__(self, cfg):
        self.qwen_vl_interface = _QWen_VL_Interface(...)    # ① Qwen2.5-VL (인코더+디코더)
        self.dino_backbone = DINOv2BackBone(...)            # ② DINOv2 (밀집 공간 피처)
        self.qformer = LayerwiseQFormer(...)                # ③ 다층 교차 어텐션 프로젝터
        self.action_model = ActionModel(...)                # ④ DiT 확산 액션 헤드

    def forward(self, batch):  # 학습: Gaussian diffusion noise prediction
        vlm_hidden = self.qwen_vl_interface(images, instructions)  # 다층 히든 스테이트
        dino_tokens = self.dino_backbone(images)                   # 패치 토큰
        condition = self.qformer(vlm_hidden, dino_tokens)          # 교차 어텐션 집계
        loss = self.action_model(condition, gt_actions)             # noise prediction MSE

    def predict_action(self, ...):  # 추론: DDIM 샘플링
        condition = ...  # 위와 동일
        actions = self.action_model.create_ddim(condition, steps=10, cfg_scale=...)
```

**커스텀 체크포인트 로딩**: HuggingFace 표준(`from_pretrained`)이 아닌, `baseframework.from_pretrained()`에서 직접 `config.json` + `dataset_statistics.json` + `.pt` 가중치 파일을 읽는 독자 포맷.

```python
# InternVLA/model/framework/base_framework.py
class baseframework(nn.Module):
    @classmethod
    def from_pretrained(cls, ckpt_path):
        cfg = read_model_config(ckpt_path)        # config.json 또는 .yaml
        norm_stats = load_json(ckpt_path / "dataset_statistics.json")
        model = build_framework(cfg)
        state_dict = torch.load(ckpt_path / "*.pt")
        model.load_state_dict(state_dict)
```

**X-VLA와의 핵심 차이**:


| 항목 | InternVLA-M1 | X-VLA |
|------|-------------|-------|
| VLM 백본 | Qwen2.5-VL (인코더+디코더 전체 사용) | Florence2 (인코더만 사용) |
| 시각 인코더 | DINOv2 (추가 밀집 인코더) | 없음 (VLM 자체만) |
| 피처 집계 | LayerwiseQFormer (교차 어텐션) | 직접 연결 |
| 액션 생성 | Gaussian Diffusion (DDPM/DDIM) | Flow-Matching (ODE) |
| 크로스-체현 | 주요 초점 아님 | 핵심 기능 (DomainAwareLinear, 소프트 프롬프트) |
| VLM 미세조정 | 지원 (VLM 공동학습) | 미지원 |
| 서빙 프로토콜 | WebSocket + msgpack | HTTP POST + json_numpy (FastAPI) |
| 설정 체계 | OmegaConf YAML + CLI dotlist | argparse |

### 에피소드 실행 루프

클라이언트 측 에피소드 루프 (LIBERO 기준):

```
for task_id in task_suite:
    env = create_env(task_id)
    for episode in range(num_episodes):
        obs = env.reset()
        model.reset(task_description)

        for step in range(max_steps):  # 기본 600
            image = rotate_180(obs["agentview_image"])
            raw_action = model.step(images=[image], step=step)

            # action chunking: step % chunk_size == 0일 때만 서버 쿼리
            # 나머지 스텝은 캐시된 action chunk에서 인덱싱

            action = unnormalize(raw_action)
            obs, reward, done, info = env.step(action)

            if done or info["success"]:
                break
```

**Action Chunking**: `step % action_chunk_size == 0`일 때만 WebSocket 서버에 쿼리. 나머지 스텝은 이전 응답의 action chunk (`future_action_window_size + 1`)에서 인덱싱. SimplerEnv에서는 action chunking 대신 매 스텝 쿼리 + `AdaptiveEnsembler`로 시간 스무딩.

### 통신 구조

**WebSocket + msgpack (비동기 서버, 동기 클라이언트)**:

```
┌─────────────────┐    msgpack/WebSocket     ┌──────────────────┐
│ eval_libero.py   │ ←────────────────────→  │ server_policy_M1  │
│ (M1Inference)    │                          │ (InternVLA_M1)    │
│                  │  type: "infer"           │                   │
│  client.infer()  │ ───→ batch_images,      │  predict_action() │
│  (sync recv)     │      instructions,       │  (GPU, DDIM)      │
│                  │      unnorm_key          │                   │
│                  │ ←─── normalized_actions  │                   │
└─────────────────┘                          └──────────────────┘
```

**프로토콜 상세**:
- 서버: `websockets` 라이브러리, asyncio 기반. 메시지 라우팅: `ping`/`infer`/`reset` 3가지 타입
- 클라이언트: `websockets.sync.client` (동기). `_wait_for_server()`로 600초 타임아웃 재시도
- 직렬화: 커스텀 `msgpack_numpy` — ndarray를 `{__ndarray__: True, data: bytes, dtype, shape}`로 변환. pickle 대비 ~4배 빠름
- 응답 형식: `{"status": "ok", "type": "infer", "data": {"normalized_actions": ndarray}}`
- **정규화는 서버에서, 역정규화는 클라이언트에서**: 서버는 [-1, 1] 범위의 정규화된 액션을 반환, 클라이언트가 `dataset_statistics.json`에서 통계를 읽어 역정규화

**starVLA와의 차이**: starVLA는 이 프로토콜을 그대로 계승했으며, 동일한 `websocket_policy_server.py`/`websocket_policy_client.py`/`msgpack_numpy.py`를 사용. starVLA의 확장점은 서버가 `FRAMEWORK_REGISTRY`를 통해 다양한 모델을 로드할 수 있도록 제네릭화한 것.

### 환경 격리

**격리 없음.** 평가 스크립트가 벤치마크 라이브러리(`libero`, `simpler_env`)를 직접 import. Docker나 프로세스 격리 없음. 이는 X-VLA와 동일한 패턴.

### 설정/CLI

**OmegaConf YAML + CLI dotlist 오버라이드**:

```yaml
# InternVLA/config/training/internvla_cotrain_oxe.yaml
framework:
  name: InternVLA-M1
  vlm:
    name: Qwen2.5-VL-3B-Instruct
  dino:
    model: dinov2_vits14
  qformer:
    num_query_tokens: 64
    start_layer: 36
    end_layer: 37
  action_model:
    type: DiT-B
    action_dim: 7
    future_action_window_size: 15

trainer:
  max_steps: 100000
  learning_rate:
    base: 1e-5
    action_model: 1e-4
  freeze_modules: ""
  loss_scale:
    vlm: 0.1
```

CLI에서 `--key=value` 형식의 dotlist로 YAML 값을 오버라이드:
```bash
accelerate launch train_internvla_cotrain.py \
    --config InternVLA/config/training/internvla_cotrain_oxe.yaml \
    --trainer.max_steps=50000 \
    --trainer.learning_rate.base=2e-5
```

`normalize_dotlist_args()`가 `--key value` 형식도 `key=value`로 변환하여 OmegaConf에 전달. Hydra는 사용하지 않으나 OmegaConf의 merge 기능을 활용.

### 결과 수집

- **Wandb**: 학습 중 `action_loss`, `vlm_loss`, `grad_angle`, `eval_mse` 등을 wandb에 로깅
- **JSONL**: 학습 메트릭을 로컬 JSONL에 기록
- **평가 결과**: LIBERO 평가 시 태스크별/스위트별 success rate를 stdout에 출력. 리플레이 비디오를 MP4로 저장 가능
- **체계적 결과 DB 없음**: 평가 결과를 구조화된 형식(JSON/DB)으로 저장하는 기능 없음

## 프로젝트별 심층 분석

### ① Dual-Encoder 아키텍처: Qwen2.5-VL + DINOv2

InternVLA-M1의 가장 특징적인 설계 — VLM과 별도의 밀집 시각 인코더를 병용:

- **Qwen2.5-VL**: 언어+시각 통합 이해. 전체 모델(인코더+디코더)을 사용하며, 중간 히든 스테이트를 추출하여 액션 조건으로 활용. `chat_with_M1()`로 언어 생성도 가능 (Dual-system)
- **DINOv2**: 공간적으로 밀집된 패치 토큰 제공. VLM이 언어-시각 정렬에 특화된 반면, DINOv2는 로봇 조작에 필요한 fine-grained 공간 정보를 보완
- **멀티뷰 처리**: DINOv2는 `ThreadPoolExecutor`로 다중 뷰 이미지를 병렬 전처리

```python
# InternVLA/model/framework/M1.py — forward() 핵심 흐름
vlm_output = self.qwen_vl_interface(images, instructions)
# vlm_output.hidden_states: 전 레이어의 히든 스테이트 리스트

dino_output = self.dino_backbone(images)
# dino_output: [B, num_patches, dino_dim] 패치 토큰

condition = self.qformer(vlm_output.hidden_states, dino_output)
# LayerwiseQFormer가 두 인코더의 출력을 교차 어텐션으로 융합
```

### ② LayerwiseQFormer: 다층 교차 어텐션 프로젝터

VLM의 서로 다른 레이어에서 추출한 히든 스테이트를 독립적으로 처리하는 프로젝터:

```python
# InternVLA/model/modules/projector/QFormer.py
class LayerwiseQFormer(nn.Module):
    def __init__(self, cfg):
        self.query_tokens = nn.Parameter(torch.randn(1, 64, hidden_dim))  # 64개 학습 가능 쿼리
        # 레이어별 독립 CrossAttentionBlock
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads)
            for _ in range(end_layer - start_layer)
        ])

    def forward(self, vlm_hidden_states, dino_tokens):
        queries = self.query_tokens.expand(B, -1, -1)
        for i, (layer_hidden, cross_attn) in enumerate(
            zip(vlm_hidden_states[start:end], self.cross_attn_layers)
        ):
            kv = torch.cat([layer_hidden, dino_tokens], dim=1)  # VLM + DINO 결합
            queries = cross_attn(queries, kv)                    # 교차 어텐션
        return queries  # [B, 64, hidden_dim] → 액션 모델 조건
```

**설계 의의**: VLM의 낮은 레이어(공간 정보 풍부)와 높은 레이어(의미 정보 풍부)를 동시에 활용. 기본 설정은 레이어 36-37만 사용하지만 확장 가능. X-VLA의 단순 연결(concatenation)보다 정보 압축이 효율적.

### ③ Gaussian Diffusion + DDIM + Optional CFG

X-VLA의 Flow-Matching과 대비되는 Gaussian Diffusion 방식:

```python
# InternVLA/model/modules/action_model/DiTActionHeader.py
class ActionModel(nn.Module):
    def __init__(self, cfg):
        self.net = DiT(...)                      # Transformer-based denoiser
        self.scheduler = GaussianDiffusion(...)   # DDPM 스케줄러

    def forward(self, condition, gt_actions):     # 학습
        noise = torch.randn_like(gt_actions)
        t = torch.randint(0, T, (B,))
        noisy_actions = self.scheduler.q_sample(gt_actions, t, noise)
        predicted_noise = self.net(noisy_actions, t, condition)
        return F.mse_loss(predicted_noise, noise)  # noise prediction 목표

    def create_ddim(self, condition, steps=10, cfg_scale=None):  # 추론
        x_t = torch.randn(B, chunk_size, action_dim)
        for t in reversed_schedule(steps):        # DDIM 역과정
            pred_noise = self.net(x_t, t, condition)
            if cfg_scale and cfg_scale > 1:       # Classifier-Free Guidance
                uncond_noise = self.net(x_t, t, null_condition)
                pred_noise = uncond_noise + cfg_scale * (pred_noise - uncond_noise)
            x_t = ddim_step(x_t, pred_noise, t)
        return x_t
```

**DiT 변형**: S(depth=6, dim=384, heads=4), B(depth=12, dim=768, heads=12), L(depth=24, dim=1024, heads=16). 기본 설정은 DiT-B.

**CFG (Classifier-Free Guidance)**: SimplerEnv 인터페이스에서 `cfg_scale=1.5`로 사용. LIBERO에서는 미사용. CFG는 조건(instruction) 부합도를 강화하는 효과.

### ④ VLA + VLM 공동학습

단일 모델로 액션 예측과 언어 생성을 동시에 학습:

```python
# InternVLA/training/train_internvla_cotrain.py — VLAMTrainer._train_step()
def _train_step(self, vla_batch, vlm_batch):
    # 1) VLA forward + backward
    action_loss = model.forward(vla_batch)   # diffusion loss
    accelerator.backward(action_loss)

    # 2) VLM forward + backward (gradient 누적)
    vlm_loss = model.qwen_vl_interface.forward_vlm(vlm_batch)  # language loss
    scaled_vlm_loss = vlm_loss * cfg.trainer.loss_scale.vlm     # 기본 0.1
    accelerator.backward(scaled_vlm_loss)

    # 3) 단일 optimizer step (두 loss의 gradient 합산)
    optimizer.step()
    optimizer.zero_grad()
```

**VLM 데이터**: LLaVA JSON 형식 — 이미지/비디오 + 대화형 QA 쌍. 18개 VLM 데이터셋 지원. `loss_scale.vlm=0.1`로 VLM loss를 축소하여 액션 학습이 주도.

### ⑤ Repeated Diffusion Training Trick

확산 모델 학습 효율성을 위한 트릭 (CogACT 참조):

```python
# InternVLA/training/train_internvla.py
# 각 배치의 동일 샘플을 N번 반복하여 서로 다른 노이즈 레벨로 학습
repeat_times = cfg.trainer.get("repeat_times", 4)
batch = repeat_batch(batch, repeat_times)  # 동일 (image, action) 쌍을
                                            # 4-8번 복제 → 각각 다른 t에서 노이즈 추가
```

동일한 관측-액션 쌍에 대해 서로 다른 확산 타임스텝(노이즈 레벨)을 동시에 학습함으로써, 배치 다양성 대비 확산 학습 수렴 속도를 개선.

### ⑥ Per-Module Learning Rate

모듈별 차등 학습률 지원:

```python
# InternVLA/training/trainer_utils/metrics.py — build_param_lr_groups()
# YAML에서 모듈 경로별 학습률 지정:
#   learning_rate:
#     base: 1e-5           # 기본 (VLM 백본 등)
#     action_model: 1e-4   # 액션 헤드 10배 높게
#     qformer: 5e-5        # 프로젝터 5배 높게

param_groups = build_param_lr_groups(model, cfg)
# → [{"params": action_model.parameters(), "lr": 1e-4},
#    {"params": qformer.parameters(), "lr": 5e-5},
#    {"params": remaining.parameters(), "lr": 1e-5}]
```

사전학습된 VLM 백본은 낮은 학습률로, 새로 초기화된 액션 헤드/프로젝터는 높은 학습률로 학습. X-VLA의 4-group 옵티마이저와 유사한 접근.

### ⑦ starVLA와의 관계

InternVLA-M1은 starVLA의 **upstream** 프로젝트:

| 항목 | InternVLA-M1 (upstream) | starVLA (downstream fork) |
|------|------------------------|---------------------------|
| 프레임워크 디스패치 | `if name == "InternVLA-M1"` | `FRAMEWORK_REGISTRY` 데코레이터 |
| VLM 백본 | Qwen2.5-VL만 | Qwen2.5-VL + InternVL |
| 액션 헤드 | DiT (Gaussian Diffusion)만 | GR00T, Fast, OFT, DiT 등 다수 |
| 벤치마크 수 | 2개 (LIBERO, SimplerEnv) | 6개+ (LIBERO, SimplerEnv, BEHAVIOR, Calvin, RoboCasa, RoboTwin) |
| 코드 성숙도 | 연구 프로토타입 | 프레임워크 지향 리팩터링 |

starVLA가 추가한 핵심 확장:
- `FRAMEWORK_REGISTRY`/`VLM_REGISTRY`/`ACTION_MODEL_REGISTRY` 레지스트리 패턴 도입
- 다수 벤치마크 인터페이스 추가
- `eval_protocol.md` 평가 아키텍처 문서화
- 서버를 프레임워크-독립적으로 제네릭화

## 코드 품질/성숙도

| 항목 | 평가 |
|------|------|
| 타입 힌트 | **부분적**. 인터페이스 파일에 `Optional`, `Dict` 등 사용. 내부 모듈은 거의 없음 |
| 테스트 | **없음**. 단위 테스트, 통합 테스트 모두 부재 |
| CI/CD | **없음**. GitHub Actions 등 미설정 |
| 문서화 | **최소**. README에 기본 사용법. 코드 내 docstring은 산발적 |
| 코드 정리 | **연구 코드 수준**. `import ipdb; ipdb.set_trace()` 주석 잔존, 중복 import, `read_mode_config` 오타 함수명 |
| 에러 처리 | **기본적**. `try/except`는 체크포인트 로딩 등 일부에만. 분산 학습에서 `dist.get_rank == 0` (호출 누락 버그) |
| 모듈성 | **중간**. 4-컴포넌트 분리는 명확하나, `build_framework()`의 if-based 디스패치로 확장성 제한 |

**총평**: 전형적인 연구 프로토타입. starVLA가 이를 fork하여 프레임워크 수준으로 리팩터링한 이유가 명확히 보임. 특히 레지스트리 패턴 부재와 단일 프레임워크 하드코딩이 확장의 가장 큰 병목.

## 시사점

### 채택할 패턴

1. **WebSocket + msgpack 프로토콜 검증**: InternVLA-M1의 프로덕션 배포가 WebSocket + msgpack을 사용하며, starVLA도 이를 계승. 우리 Protocol 설계(WebSocket + msgpack binary)가 VLA 생태계의 사실상 표준임을 재확인.
2. **Dual-Encoder 개념의 일반화**: VLM + 별도 시각 인코더 병용은 다양한 VLA에서 채택 중. 이는 `PredictModelServer.predict()` 내부의 모델 구현 세부사항이며, 우리 프레임워크가 강제하지 않는 영역. 모델의 인코더 구성을 기술하는 메타데이터/설정 필드로 추적 가능.
3. **LayerwiseQFormer의 교차 어텐션 패턴**: "어떤 레이어에서 피처를 추출하는가"가 모델 성능에 중요. 이를 벤치마크 결과와 연관지어 분석할 수 있는 메타데이터 수집.
4. **Action Chunking / Ensemble → PredictModelServer 내장 기능**: InternVLA-M1은 벤치마크에 따라 다른 후처리 전략(LIBERO=chunking, SimplerEnv=ensemble)을 사용. 우리 설계에서는 이를 별도 `ActionPostProcessor`가 아닌 `PredictModelServer`의 내장 파라미터로 처리한다:
   - InternVLA-M1의 action chunking → `PredictModelServer.chunk_size` (e.g. `chunk_size=16`)
   - InternVLA-M1의 `AdaptiveEnsembler` → `PredictModelServer.action_ensemble` (`"average"` | `"newest"` | callable)
   - InternVLA-M1의 DDIM 디노이징 → `predict()` 내부의 모델 구현 세부사항
   - InternVLA-M1의 Gaussian Diffusion → `predict()` 내부의 모델 구현 세부사항
5. **`predict_action() → normalized_actions` 매핑**: InternVLA-M1의 `predict_action()`이 정규화된 액션을 반환하는 패턴은 우리 `PredictModelServer.predict() → {"actions": ndarray}`에 대응. 정규화/역정규화 위치 분리(서버=정규화, 클라이언트=역정규화)도 동일 원칙.
6. **Per-module LR + Freeze, Repeated Diffusion Trick**: 학습 세부사항으로 우리 평가 프레임워크의 범위 밖. 다만 학습 설정이 모델 성능에 미치는 영향을 분석하려면, 이러한 하이퍼파라미터를 벤치마크 결과 메타데이터에 포함할 수 있음.
7. **OmegaConf dotlist 오버라이드**: Hydra 없이도 YAML + CLI 오버라이드 가능. 가벼운 설정 체계로 참고 가치.

### 회피할 패턴

1. **if-based 프레임워크 디스패치**: `build_framework()`의 if-elif 체인은 starVLA가 레지스트리로 대체한 것처럼, 확장에 취약. 우리는 처음부터 레지스트리/플러그인 패턴 사용
2. **커스텀 체크포인트 포맷**: HuggingFace `from_pretrained()` 표준을 벗어난 독자 포맷은 호환성 문제. 우리 프레임워크에서 모델 로딩은 가능한 HF 표준을 따르되, 비표준도 어댑터로 지원
3. **벤치마크별 M1Inference 중복**: 같은 이름의 클래스가 파일별로 별도 구현. 우리는 공통 BaseInference ABC + 벤치마크별 mixin으로 해결
4. **`dist.get_rank == 0` 호출 누락 버그**: 분산 학습 코드에서 `()` 누락으로 항상 truthy. 이런 패턴은 정적 분석/린트로 잡아야 함
5. **평가 결과의 비구조화 출력**: stdout 출력만으로는 결과 비교/추적이 불가능. 구조화된 결과 저장 필수

### 열린 질문

1. **Gaussian Diffusion vs Flow-Matching**: InternVLA-M1(DDPM/DDIM)과 X-VLA(Flow-Matching)의 성능 차이는? 우리 프레임워크에서 동일 벤치마크/조건으로 비교 평가 필요. 두 방식 모두 `PredictModelServer.predict()` 내부 구현이므로 프레임워크 수준에서는 투명하게 비교 가능.
2. **Dual-Encoder(VLM+DINO) vs Single-Encoder**: DINOv2 추가의 실질적 성능 기여도는? ablation 기준이 될 수 있음.
3. **CFG Scale의 벤치마크별 최적값**: SimplerEnv에서 1.5, LIBERO에서 미사용 — 벤치마크 특성에 따른 최적 CFG 탐색이 필요.
4. **정규화 전략 차이**: LIBERO는 min/max, SimplerEnv는 q01/q99 분위수 사용. 같은 모델이 벤치마크에 따라 다른 정규화를 적용하는 것이 평가 공정성에 미치는 영향.
5. **Action Ensemble 전략의 벤치마크별 최적화**: InternVLA-M1이 LIBERO에서는 chunking(`chunk_size > 1`, `action_ensemble="newest"`), SimplerEnv에서는 `AdaptiveEnsembler`(`action_ensemble="average"` 또는 callable)를 사용하는 것처럼, `PredictModelServer`의 `chunk_size`와 `action_ensemble` 파라미터의 벤치마크별 최적 조합을 체계적으로 탐색할 수 있는가?