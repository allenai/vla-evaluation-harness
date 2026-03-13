# lm-evaluation-harness — Reference Survey

## 기본 정보

| 항목 | 내용 |
|------|------|
| Repository | https://github.com/EleutherAI/lm-evaluation-harness |
| Stars / Last Commit | ~8k+ stars / 2026-02-15 |
| 주요 목적 | LLM 평가를 위한 통합 프레임워크. 160+ task 디렉토리, 13,000+ YAML 설정, 25+ 모델 백엔드 지원 |
| 우리 프로젝트에서의 참고 포인트 | 01_PROPOSAL.md: "Task/벤치마크 추상화, 확장 가능한 평가 프레임워크 설계의 교과서적 사례" |

## 프로젝트 구조

```
lm_eval/
├── api/                    # 핵심 추상화 계층
│   ├── task.py             # Task ABC + ConfigurableTask (1800 lines)
│   ├── model.py            # LM ABC + TemplateLM + CachingLM (543 lines)
│   ├── registry.py         # Generic Registry[T] — thread-safe, lazy loading (720 lines)
│   ├── group.py            # Group dataclass — 계층적 task 컨테이너 (395 lines)
│   ├── instance.py         # Instance dataclass — 모델 요청 단위 (39 lines)
│   ├── filter.py           # Filter ABC + FilterEnsemble (57 lines)
│   ├── metrics.py          # 메트릭 함수들
│   ├── samplers.py         # Few-shot 샘플러
│   └── utils.py
├── config/                 # 설정 데이터클래스
│   ├── task.py             # TaskConfig (30+ 필드), FewshotConfig
│   ├── group.py            # GroupConfig, AggMetricConfig
│   └── evaluate_config.py
├── _cli/                   # CLI 서브커맨드
│   ├── harness.py          # 진입점 (lm-eval 명령)
│   ├── run.py              # 평가 실행 (471 lines)
│   ├── ls.py               # task 목록 조회
│   └── validate.py         # task 설정 검증
├── evaluator.py            # 핵심 평가 루프 (701 lines)
├── evaluator_utils.py      # 결과 후처리 (515 lines)
├── result_schema.py        # TypedDict 기반 결과 스키마 (210 lines)
├── tasks/                  # 160+ task 디렉토리, 13,000+ YAML 설정
│   ├── manager.py          # TaskManager — task 검색/로딩 (368 lines)
│   ├── _factory.py         # TaskFactory — task 인스턴스 생성 (286 lines)
│   ├── _index.py           # TaskIndex — YAML 스캔/인덱싱 (200 lines)
│   ├── _yaml_loader.py     # YAML 파서 (커스텀 !function 태그)
│   └── hellaswag/          # 예시: hellaswag.yaml (23 lines) + utils.py
├── models/                 # 25+ 모델 백엔드
│   ├── huggingface.py      # HFLM — 주력 구현 (1595 lines)
│   ├── vllm_causallms.py   # vLLM 백엔드
│   ├── openai_completions.py
│   ├── anthropic_llms.py
│   ├── hf_vlms.py          # 비전-언어 모델
│   └── ... (20+ more)
├── filters/                # 후처리 필터
├── loggers/                # 결과 추적
│   ├── evaluation_tracker.py  # EvaluationTracker + HF Hub 연동
│   └── wandb_logger.py
├── caching/                # SQLite 기반 응답 캐싱
└── decontamination/        # 데이터 오염 검출
```

**코드 규모**: `lm_eval/` 하위 Python 파일 총 ~22,000 lines. Task YAML 13,000+ 파일.

## 핵심 분석

### 벤치마크 추상화

**3단계 추상화 계층**이 핵심 설계:

1. **Task ABC** (`api/task.py`): 추상 메서드 5개 — `has_training_docs()`, `has_validation_docs()`, `has_test_docs()`, `doc_to_text(doc)`, `doc_to_target(doc)`. 데이터셋 로딩(`download()`), 요청 생성(`build_all_requests()`), 결과 처리(`process_results()`)의 전체 생명주기를 정의.

2. **ConfigurableTask** (`api/task.py:618`): Task를 상속하되, **YAML 설정만으로 모든 추상 메서드를 자동 구현**. `TaskConfig` 데이터클래스(30+ 필드)를 읽어 `doc_to_text`는 Jinja2 템플릿으로, `metric_list`는 레지스트리 조회로 해결. 대부분의 task가 이 클래스를 직접 사용.

3. **YAML 설정 파일**: 최하위 레이어. 23줄짜리 `hellaswag.yaml`이 하나의 완전한 벤치마크를 정의:

```yaml
task: hellaswag
dataset_path: Rowan/hellaswag      # HuggingFace Hub 데이터셋
output_type: multiple_choice        # loglikelihood | generate_until | ...
training_split: train
validation_split: validation
process_docs: !function utils.process_docs  # 커스텀 전처리
doc_to_text: "{{query}}"           # Jinja2 템플릿
doc_to_target: "{{label}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
```

**Group 계층**: `Group` dataclass가 Task/sub-Group을 트리 구조로 조직. MMLU처럼 57개 하위 task를 subject → category → overall로 계층적으로 집계. `aggregate()` 메서드가 leaf task의 메트릭을 weighted mean으로 합산.

**output_type 4가지**: `loglikelihood`, `loglikelihood_rolling`, `generate_until`, `multiple_choice`. 이것이 모델에 전달되는 요청 유형을 결정하며, Task와 Model 사이의 계약(contract)을 형성.

### 모델 통합 패턴

**LM ABC** (`api/model.py`): 추상 메서드 3개만 요구:

```python
class LM(abc.ABC):
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]: ...
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]: ...
    def generate_until(self, requests: list[Instance]) -> list[str]: ...
```

**TemplateLM 중간 계층** (`api/model.py`): LM ABC와 구체 구현 사이의 중간 추상 클래스. 토크나이저 기반 공용 로직을 제공 — `_encode_pair(context, continuation)`, `_loglikelihood_tokens()` 등. HuggingFace, vLLM 등 토크나이저를 사용하는 모든 모델이 이 클래스를 상속하여 중복 코드를 제거.

**CachingLM 데코레이터** (`api/model.py`): 투명한 SQLite 기반 응답 캐싱. 임의의 LM 인스턴스를 감싸서 3개 추상 메서드를 인터셉트 — 캐시 히트 시 모델 호출 없이 반환, 미스 시 실제 모델 호출 후 캐시에 저장. 모델 코드 수정 없이 데코레이터 패턴으로 적용.

**모델 등록/생성**: `@register_model("hf-auto", "hf", "huggingface")` 데코레이터로 모델 클래스를 레지스트리에 등록. `create_from_arg_string(arg_string)` 팩토리 메서드가 CLI 인자 문자열(`"pretrained=gpt2,dtype=float16"`)을 파싱하여 모델 인스턴스를 생성. 25+ 모델 백엔드: HuggingFace(HFLM, 1595 lines), vLLM, SGLang, OpenAI, Anthropic, Megatron-LM, GGUF, HF VLMs, HF AudioLM 등.

**분산 지원**: LM ABC에 `rank`, `world_size`, `accelerator` 속성이 내장. `all_gather()`, `gather_object()`, `barrier()` 메서드로 멀티-GPU/멀티-노드 평가를 지원. HFLM은 HuggingFace Accelerate를 통한 data-parallel 지원, 자동 배치 크기 탐색, PEFT/LoRA, 양자화(GPTQ, GGUF) 지원.

### 에피소드 실행 루프

lm-evaluation-harness는 **에피소드 기반이 아닌 배치 요청 모델**. 대화형 환경과의 상호작용이 없으므로 "에피소드"라는 개념 자체가 부재. 대신 `evaluator.py`의 `evaluate()` 함수(701 lines)가 핵심 평가 루프를 구성:

**`simple_evaluate()` → `evaluate()` 흐름**:

1. **모델 인스턴스화**: 문자열로 전달된 모델명을 레지스트리에서 조회하여 인스턴스 생성. 선택적으로 `CachingLM`으로 래핑.
2. **Task 로딩**: `TaskManager.load(task_names)` → `TaskIndex` 스캔 → `TaskFactory.build()` → `TaskDict` 반환 (`{tasks, groups, group_map}`).
3. **요청 빌딩** (lines 524-570): 모든 task에 대해 `task.build_all_requests()` 호출. 생성된 `Instance` 객체들을 `request_type`별로 그룹화.
4. **Cross-task 배칭 + 모델 디스패치** (lines 572-593): **핵심 효율성 메커니즘** — 동일한 `request_type`의 Instance를 **모든 task에 걸쳐** 하나로 모은 뒤, `getattr(lm, reqtype)(cloned_reqs)` 한 번의 호출로 처리. Task A의 `generate_until` 요청과 Task B의 `generate_until` 요청이 하나의 배치로 합쳐짐.
5. **필터 적용 + 결과 처리** (lines 597-655): 각 task에 대해 `task.apply_filters()` → 문서별 `task.process_results(doc, filtered_resps)` → 메트릭 누적.
6. **분산 수집** (lines 657-688): 멀티-랭크 환경에서 `all_gather()`로 샘플/메트릭 합산.
7. **최종 집계** (lines 689-700): `_process_results()`가 최종 메트릭, stderr, Group 집계를 계산하여 `EvalResults` 반환.

### 통신 구조

**N/A** — 모델이 동일 Python 프로세스 내에서 실행됨. 별도의 네트워크 통신 프로토콜 없음. 모든 모델 호출은 함수 호출 수준. API 기반 모델(OpenAI, Anthropic)은 각 모델 클래스 내부에서 HTTP 클라이언트를 직접 사용하지만, 이는 프레임워크 수준의 통신 추상화가 아닌 개별 모델 구현의 문제.

### 환경 격리

**N/A** — 시뮬레이션 환경 개념 자체가 부재. 정적 데이터셋에 대한 모델 추론만 수행. Docker 격리, 환경 초기화/리셋 등의 메커니즘 없음.

### 설정/CLI

**CLI 서브커맨드 패턴** (`_cli/`, `argparse` 표준 라이브러리 기반):
- `lm-eval run` — 평가 실행 (주력 커맨드). `--model hf --model_args pretrained=gpt2 --tasks hellaswag,mmlu --batch_size 16`
- `lm-eval ls` — 등록된 task/group 목록 조회
- `lm-eval validate` — YAML task 설정 검증

**설정 체계**:
- **TaskConfig** dataclass (30+ 필드): `task`, `dataset_path`, `dataset_name`, `output_type`, `doc_to_text`/`doc_to_target`/`doc_to_image`/`doc_to_audio` (Callable | str), `metric_list`, `generation_kwargs`, `repeats`, `filter_list`, `fewshot_config` 등. YAML에서 직접 매핑.
- **GroupConfig**, **AggMetricConfig**: Group 정의 및 메트릭 집계 설정.
- **FewshotConfig**: few-shot 예시 포맷팅 제어 (sampler, delimiter, context_size 등).
- **YAML 커스텀 태그**: `!function utils.process_docs` — Python callable을 YAML에서 직접 참조. `_yaml_loader.py`의 커스텀 로더가 `"module:object"` 형식으로 해석.
- **Jinja2 템플릿**: `doc_to_text: "{{query}}"` — 데이터셋 필드를 프롬프트로 변환. 조건문, 루프 등 Jinja2 전체 문법 사용 가능.

### 결과 수집

**TypedDict 기반 결과 스키마** (`result_schema.py`, 210 lines):

- **`EvalResults`** (최상위): `results` (per-task 메트릭), `groups` (Group 집계), `group_subtasks`, `configs`, `versions`, `n-shot`, `higher_is_better`, `n-samples`, `samples`, `config` (_EvalConfig), `git_hash`, `date`, env/tokenizer 메타데이터, `task_hashes`, `total_evaluation_time_seconds`. `total=False`로 모든 키가 선택적.
- **`_TaskMetrics`** (Generic[T], extra_items=T): `name`, `alias`, `sample_len` + 동적 `"metric,filter"` 키 (예: `"acc,none"`, `"acc_stderr,none"`).
- **`SampleResult`**: `doc_id`, `doc`, `target`, `arguments`, `resps`, `filtered_resps`, `filter`, `metrics`, `doc_hash`, `prompt_hash`, `target_hash`. 재현성을 위한 해시 포함.
- **`_EvalConfig`**: 모델/배치/시드/디바이스 등 실행 설정 기록.

**EvaluationTracker** (`loggers/evaluation_tracker.py`, 587 lines):
- `GeneralConfigTracker`: 모델 메타데이터 (모델명, 소스, 시스템 명령, 채팅 템플릿), 타이밍 정보 (시작/종료/총 시간) 추적.
- `EvaluationTracker`: 결과 저장소 관리. 로컬 파일 출력 + **HuggingFace Hub 자동 푸시** (결과 요약 + per-sample 상세). `save_results_aggregated(results, samples)` 메서드가 JSON/JSONL 파일 생성 후 Hub에 업로드. 공개/비공개 repo 지원, gated repo 지원.
- **W&B 통합** (`wandb_logger.py`): Weights & Biases 로깅.
- **Per-sample 로깅**: 각 문서별 모델 입출력, 필터 결과, 메트릭, 해시를 기록하여 재현성 및 오류 분석 지원.

## 프로젝트별 심층 분석

### YAML-first Task 추가 워크플로우

외부 기여자의 새 벤치마크 추가 과정이 극도로 간결:

1. `lm_eval/tasks/my_benchmark/` 디렉토리 생성
2. `my_benchmark.yaml` 작성 (23줄이면 완성): dataset_path, output_type, doc_to_text, metric_list 등 지정
3. (선택) `utils.py` — 데이터셋 전처리 함수 작성, YAML에서 `!function utils.process_docs`로 참조
4. 끝. Python 클래스 작성 불필요. `lm-eval run --tasks my_benchmark`로 즉시 실행 가능.

`TaskIndex.build(paths)`가 디렉토리를 재귀 탐색하여 YAML을 자동 발견. `_kind_of(cfg)` 패턴 매칭이 `{"task": _}` → TASK, `{"group": _}` → GROUP, `{"class": _}` → PY_TASK로 분류.

이 워크플로우가 13,000+ YAML 파일 = 160+ 벤치마크라는 거대한 생태계를 가능하게 한 핵심 요인. Python 코드를 작성할 줄 모르는 연구자도 새 벤치마크를 추가할 수 있음.

### Registry 패턴 심층 분석

`Registry[T]` (`api/registry.py`, 720 lines)는 프레임워크 전체의 컴포넌트 발견/등록 메커니즘:

- **Generic[T]**: 타입 안전한 레지스트리. `base_cls` 지정 시 등록/조회 시점에 타입 검증.
- **Thread-safe**: `threading.RLock`으로 동시 접근 보호. 특히 lazy loading 시 더블 체크 패턴 적용.
- **Lazy loading**: 등록 시 `"module:object"` 문자열이나 `EntryPoint`를 placeholder로 저장. `get()` 호출 시 `@lru_cache`로 한 번만 materialize. 사용하지 않는 모델은 import되지 않아 시작 속도 향상.
- **`freeze()`**: `MappingProxyType`으로 감싸서 불변 보장. 초기화 완료 후 실수로 수정하는 것을 방지.
- **EntryPoint 플러그인 시스템**: Python packaging의 entry_points를 통한 서드파티 모델/필터 등록. `pip install my-model-plugin` 후 자동 발견.
- **친절한 에러 메시지**: `_suggest_similar()`가 오타 시 유사한 등록명을 제안. `_build_key_error_msg()`가 사용 가능한 항목 목록을 표시.
- **6개 레지스트리 인스턴스**: `model_registry`, `filter_registry`, `aggregation_registry`, `metric_registry`, `metric_agg_registry`, `higher_is_better_registry`.

### Filter 파이프라인

모델 출력의 후처리를 체계화한 파이프라인:

- **Filter ABC** (`api/filter.py`): `apply(resps, docs) -> Iterable` 단일 추상 메서드. 응답 리스트와 문서 리스트를 받아 필터링된 응답 리스트 반환.
- **FilterEnsemble** dataclass: 여러 Filter를 체이닝. `filters` 리스트의 각 필터를 순서대로 적용. 결과를 `inst.filtered_resps[name]`에 저장 — 키가 FilterEnsemble 이름이므로, 하나의 task에 **여러 독립적인 필터 파이프라인**을 동시 적용 가능.
- **내장 필터** (`filters/`): `extraction.py` (정규식 추출), `selection.py` (argmax/argmin 선택), `transformation.py` (소문자 변환 등), `custom.py` (커스텀 함수), `decontamination.py` (오염 검출).
- **YAML에서 지정**: task YAML의 `filter_list`에서 필터 파이프라인을 선언적으로 정의.

### Cross-task 요청 배칭

효율성의 핵심 메커니즘:

```
Task A (hellaswag): generate_until 요청 100개
Task B (mmlu):      generate_until 요청 200개
Task C (truthfulqa): loglikelihood 요청 150개
─────────────────────────────────────────────
evaluator가 수집:
  generate_until 배치: 300개 (A+B 합산) → lm.generate_until() 1회 호출
  loglikelihood 배치: 150개 (C)         → lm.loglikelihood() 1회 호출
```

Task별로 개별 호출하지 않고 **request_type 기준으로 모든 task의 요청을 합산**하여 모델에 전달. GPU 활용률을 극대화하고, 배치 크기를 키울 수 있음. 결과는 각 Instance의 `resps` 필드에 기록되어 원래 task로 라우팅됨.

## 코드 품질/성숙도

| 항목 | 평가 |
|------|------|
| **타입 힌트** | 광범위. TypedDict 기반 결과 스키마, Generic[T] 레지스트리, overload 데코레이터 등 고급 타이핑 활용 |
| **테스트** | 포괄적 테스트 스위트 존재 (별도 `tests/` 디렉토리) |
| **CI** | GitHub Actions 기반 CI/CD |
| **문서화** | docstring 충실. 특히 registry.py, result_schema.py는 각 필드에 상세 주석 |
| **코드 규모** | Python ~22,000 lines + YAML 13,000+ 파일. 대규모 프로젝트 |
| **아키텍처** | 모듈 분리 우수. api/ (추상화), config/ (설정), tasks/ (데이터), models/ (구현), loggers/ (출력)이 명확히 분리 |
| **확장성** | 160+ task, 25+ model backend이 동일 프레임워크에서 동작하는 것 자체가 확장성 증명 |

## 시사점

### 채택할 패턴

1. **Registry + lazy loading → Benchmark + ModelServer 등록**: lm-eval-harness의 `Registry[T]` 패턴을 우리 프레임워크의 Benchmark/ModelServer/EpisodeRunner 등록에 적용. 특히 `"module:object"` lazy loading은 벤치마크별 무거운 의존성(OmniGibson, MuJoCo 등)을 필요 시점까지 지연시키는 데 매우 유용. `@register_model` 데코레이터 패턴은 `@register_benchmark`, `@register_model_server`로 대응.

2. **YAML-first 설정 → YAML config 시스템**: lm-eval-harness의 TaskConfig YAML이 벤치마크를 선언적으로 정의하듯, 우리 YAML config에서 벤치마크별 Docker 이미지, mode, tasks, max_steps, seed 등을 선언적으로 지정. ConfigurableTask 수준의 완전 자동화는 시뮬레이션 환경 특성상 어렵지만, 설정 레이어의 선언적 접근은 채택할 가치가 있음.

3. **ConfigurableTask → Benchmark ABC 패턴**: lm-eval-harness의 ConfigurableTask가 YAML 설정으로 Task ABC의 추상 메서드를 자동 구현하듯, 우리 Benchmark ABC(`get_tasks()`, `reset()`, `step()`, `make_obs()`, `is_done()`, `get_result()`)는 벤치마크 기여자가 구현해야 하는 최소 계약을 정의. gymnasium.Env 유사 인터페이스로 벤치마크 추가 진입 장벽을 낮춤.

4. **CLI single entry point → Orchestrator CLI**: lm-eval-harness의 `lm-eval run` 단일 진입점 패턴을 우리 Orchestrator CLI에 적용. `argparse` 서브커맨드로 평가 실행, 벤치마크 목록 조회, 설정 검증 등을 제공.

5. **Filter 파이프라인 → Benchmark.make_obs() 관찰 변환**: lm-eval-harness의 Filter가 모델 출력을 후처리하듯, 우리 프레임워크에서는 `Benchmark.make_obs()`가 환경의 raw observation을 모델 서버에 전송할 dict로 변환하는 전처리 파이프라인 역할을 담당. Action 후처리(chunking, ensemble)는 별도 Filter 계층이 아닌 PredictModelServer의 내장 `chunk_size`/`action_ensemble` 파라미터로 처리.

6. **Group 계층 + 메트릭 집계 → Result Collector**: MMLU의 subtask → category → overall 집계 구조를 우리 Result Collector의 에피소드 → 태스크 → 벤치마크 단위 메트릭 집계에 적용. LIBERO의 10개 suite, CALVIN의 chain 길이별 성공률 등을 계층적으로 조직하고 reference score와 비교.

7. **TypedDict 결과 스키마**: `EvalResults`처럼 강타입 결과 구조를 정의하여 Result Collector의 출력 포맷 일관성 보장, IDE 자동완성 지원.

8. **친절한 에러 메시지**: 오타 시 유사 이름 제안, 사용 가능 항목 나열 등은 Orchestrator CLI의 UX 관점에서 중요.

### 회피할 패턴

1. **In-process 모델 통합**: 모델이 동일 프로세스에서 실행되는 구조. 우리는 client-server 분리가 필수 (시뮬레이션 환경과 모델 서버의 의존성 격리). 이 프로젝트의 가장 큰 한계이자 우리와의 근본적 차이.

2. **정적 데이터셋 평가**: 고정된 데이터셋에 대한 일회성 추론. 우리는 환경과의 **상호작용적(interactive)** 평가가 필요 — 모델의 행동이 환경 상태를 변경하고, 다음 관측에 영향.

3. **텍스트 전용 output_type**: `loglikelihood`, `generate_until` 등 텍스트 기반 요청 유형. 우리는 **연속 행동 공간**(action: float array)이 출력이므로 완전히 다른 계약이 필요.

4. **13,000+ YAML 파일의 관리 복잡성**: 거대한 task 생태계가 장점이지만, 동시에 일관성 유지가 어려움. YAML 스키마 검증(`validate` 커맨드)의 존재 자체가 이 문제를 반증.

### 열린 질문

1. **Cross-task 배칭의 실시간 환경 적용**: 정적 데이터셋에서는 모든 요청을 미리 수집하여 배칭할 수 있지만, 실시간 환경에서는 각 step마다 즉시 응답이 필요. 우리 설계에서는 BatchPredictModelServer가 multi-session observation을 모아 배치 추론하는 방식으로 유사한 효율성을 달성. 단일 session에서는 PredictModelServer로 충분.

2. **Group 계층의 벤치마크 적용**: MMLU의 57개 subtask → category → overall 구조를 우리 Result Collector에서 어떻게 매핑할 것인가? 예: LIBERO_10 suite를 각각 Task로, LIBERO_SPATIAL/LIBERO_OBJECT 등을 Group으로 조직하여 계층적 집계?

3. **YAML-first의 한계**: 시뮬레이션 환경은 `dataset_path`로 데이터셋을 로딩하는 것이 아니라, Docker 이미지를 실행하고 환경을 초기화해야 함. 우리 YAML config에서 벤치마크별 Docker 이미지, mode, max_steps 등을 선언적으로 지정하되, 환경 초기화 로직은 Benchmark ABC 구현에 위임하는 현재 설계가 적절한 균형점인지?
