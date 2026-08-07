# Qwen3-VL-4B 本地 Canonical r12 全链路与热点审计

- **运行日期**：2026-08-07
- **隔离 worktree**：`D:\tmp\robata-local-qwen-e2e-20260806`
- **运行**：`canonical-qwen-full-r12-20260806`
- **状态**：`INCOMPLETE / RUN_NOT_COMPLETABLE`
- **证据等级**：`LOCAL_CONFORMANCE`
- **生产资格**：`false`
- **Canonical authority**：`false`

> 这是当前硬化代码的主审计记录。它验证了完整 MCAP source、单 worker 本地 Qwen QA、lineage、scheduler、trace、GPU 和存储审计；它不宣称生产链路完成或生产就绪。

## 1. 结论摘要

r12 使用完整 `sample-medium.mcap` 和独立、初始为空的 endpoint durable idempotency 状态。运行后 endpoint 状态目录写入 51 条新请求记录，因此以下 provider latency 是干净的单 worker 本地 Qwen 样本，而不是旧状态 replay。

- source：`ADMITTED`；6 路 camera ledgers 完整。
- scheduling：205/205 work items 和 attempts `SUCCEEDED`。
- inference：`QA_COARSE` 41 个 terminal、`QA_DENSE` 10 个 terminal；51/51 `SUCCEEDED` 且 `output_valid=true`。
- evidence：8 张 lineage 表各 51 行，`lineage_complete=true`。
- reduction：`0:cam_02`、`0:cam_03`、`0:cam_05`、`0:cam_06` 最终为 `DEGRADED`；fail-closed 质量门阻断。
- publication：`NOT_COMMITTED`；primary run 仍 `RUNNING`，completion/outbox/delivery/finalization 未闭合。
- stream：调度参与，但真实 Qwen stream artifacts 为 0，route 为 `FIXTURE_ONLY`。

所以 `ok=false` 是预期且正确的质量边界，不是审计器崩溃。

## 2. 模型和 checkpoint 身份

| 项目 | 值 |
|---|---|
| 模型 | `Qwen3-VL-4B-Instruct` |
| provider | `local-huggingface` |
| model version | `local-2026-08-06` |
| endpoint | `http://127.0.0.1:8101` |
| endpoint health | `READY`, `loaded=true`, `concurrency=1` |
| adapter | `local-hf-loopback-adapter-v1` |
| execution mode | `single-worker-production-policy-local-conformance` |
| manifest version | `local-hf-checkpoint-manifest-v1` |
| manifest SHA-256 | `1f7293b2629473f0240c8675025e1402da4306f05cc9026adf4c801f20f99f10` |
| included files | 12 |
| HF revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| parser version | `local-hf-compact-provider-claim-v1-088bf2b9e39d` |
| mapping | `genrobot-observed-v0.json`, `UNAPPROVED`（仅显式 allow 用于本地 conformance） |

checkpoint manifest、HF revision、model version、adapter 和 parser contract 都进入本次 binding/health 预检；r8 文档中“尚未绑定 checkpoint”的旧缺口不适用于 r12。

## 3. Source admission

| 项目 | 值 |
|---|---:|
| source | `D:\Github\Robata\data\source\sample-medium.mcap` |
| size | 130,303,923 bytes |
| SHA-256 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| MCAP messages | 16,210 |
| channels | 17 |
| native span | 40.890455 s |
| admitted duration | 40.833513001 s |
| admitted timing frames | 7,350（6 cameras × 1,225） |
| quality observations | 82 per camera |
| camera ledgers | 6 |
| requested max duration | 41 s |
| window limited | `false` |

完整源已通过持久化 `media-quality-report-v1` 准入。模型输入不是全部 7,350 timing frames：本运行产生 306 个 image inputs（coarse 246、dense 60），其余帧参与窗口、质量和 source ledger 语义。

## 4. QA、lineage 和质量门

### 4.1 Terminal 与 lineage

| stage | terminal | status |
|---|---:|---|
| `QA_COARSE` | 41 | 41 `SUCCEEDED`, valid |
| `QA_DENSE` | 10 | 10 `SUCCEEDED`, valid |
| **合计** | **51** | **51/51 成功** |

以下八段 lineage 均为 51 行：

```text
inference_intents
model_inference_terminals
raw_provider_artifacts
raw_provider_responses
parsed_provider_claims
inference_attempt_selections
selected_attempt_outputs
enriched_provider_outputs
```

raw media type 为 `application/json`（51/51）；parser version 51/51 相同为 `local-hf-compact-provider-claim-v1-088bf2b9e39d`。严格 compact decoder 不接受未知标签、额外文本或宽松 alias。

### 4.2 Provider latency（独立冷状态）

| stage | n | min | P50 | P95 | max | sum | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `QA_COARSE` | 41 | 1,926 ms | 2,150 ms | 11,703 ms | 12,252 ms | 213,415 ms | 5,205.244 ms |
| `QA_DENSE` | 10 | 2,017 ms | 2,125 ms | 11,654 ms | 11,654 ms | 35,073 ms | 3,507.300 ms |
| **total** | **51** |  |  |  |  | **248,488 ms** |  |

P50 约 2.1 秒，但长尾达到 11–12 秒；单 worker 下不能用 P50 单独做容量承诺。

### 4.3 Dense reduction

策略为 `severity-worst-then-earliest-v1`，严重度 `GOOD < DEGRADED < UNKNOWN < UNUSABLE`。输入和最终结果：

| coordinate | observations | reduced |
|---|---|---|
| `0:cam_01` | `GOOD`, `GOOD` | `GOOD` |
| `0:cam_02` | `DEGRADED`, `GOOD`, `GOOD` | **`DEGRADED`** |
| `0:cam_03` | `DEGRADED`, `GOOD` | **`DEGRADED`** |
| `0:cam_04` | `GOOD`, `GOOD` | `GOOD` |
| `0:cam_05` | `DEGRADED`, `GOOD`, `GOOD` | **`DEGRADED`** |
| `0:cam_06` | `DEGRADED`, `GOOD` | **`DEGRADED`** |

`DEGRADED` 没有被重写成 `GOOD`；因此错误边界为 `REDUCTION`，错误码 `RUN_NOT_COMPLETABLE`。这是质量语义阻断，不是 reduction 性能问题。

## 5. Scheduler / stream / publication

| 项目 | 值 |
|---|---:|
| expected windows | 41 |
| work items / attempts | 205 / 205，全部成功 |
| stream window results | 41 |
| stream evidence commits | 41 |
| stream artifacts | 410 |
| fixture intent / accepted-call artifacts | 41 / 41 |
| real Qwen stream artifacts | 0 |
| route | `FIXTURE_ONLY` |
| stream delivery outbox | 41，全部 `PENDING` |
| outbox deliveries | 41，全部 `PENDING` |
| recording finalizations | 0 |
| primary runs | 1，`RUNNING` |
| primary completions / detailed results | 0 / 0 |
| primary outbox / deliveries | 0 / 0 |
| publication | `NOT_COMMITTED` |

`QWEN_EVENT_PROPOSAL` scheduler item 成功只代表 fixture scheduler ledger 成功，不能当作真实 event-proposal 模型证据。

## 6. Trace 和性能热点

### 6.1 总体资源

| 项目 | r12 |
|---|---:|
| elapsed wall | 1,127.6320984 s（约 18 分 47.6 秒） |
| admitted video | 40.833513001 s |
| real-time factor | 27.615× slower than real time |
| throughput | 0.03621 video-hour / wall-hour |
| 24h 线性外推 | 0.869 video-hours/day |
| process CPU | 915.9375 s |
| RSS | 603.21 MiB |
| read I/O | 4.267 GiB |
| write I/O | 573.17 MiB |
| trace spans | 13,105 |
| unclassified spans | 0 |

### 6.2 Stage wall union

Stage union 有嵌套，不能相加：

| stage | spans | union | elapsed 占比 |
|---|---:|---:|---:|
| `ORCHESTRATION` | 4 | 1,123.406 s | 99.63% |
| `SOURCE` | 22 | 854.232 s | 75.75% |
| `SCHEDULING` | 11,636 | 7.199 s | 0.64% |
| `INFERENCE` | 158 | 276.373 s | 24.51% |
| `EVIDENCE` | 1,282 | 8.805 s | 0.78% |
| `REDUCTION` | 2 | 29.5 ms | 0.003% |
| `PUBLICATION` | 1 | 4.1 ms | <0.001% |

### 6.3 细粒度热点

| span | elapsed | 解释 |
|---|---:|---|
| `source.prepare` | 844.470 s | cold source preparation 主体 |
| `source.media.package_binding` | 746.188 s | `mode=EXPORT_TIME`，最大单一热点 |
| `source.stream.capture_publish` | 97.426 s | 六路 capture publish |
| `source.media.decode` | 37.738 s | media decode |
| `source.media.encode` | 36.629 s | evidence PNG encode |
| `source.media.selection` | 22.896 s | 六路 frame selection |
| `inference.pipeline` | 276.373 s | 51 个串行模型调用和外围工作 |

source span 和子 span 嵌套；不要把表中秒数简单求和。r8 历史热路径的 source union 约 87.227 s、package binding 约 12.965 s，模式为 `REPLAY_DECODE`。这证明冷导出与热复用是两种不同的容量条件；r12 的首次导入瓶颈不是 SQLite 或 reduction。

provider latency 248.488 s 占总 elapsed 22.04%，占 inference union 89.91%；inference 外围约 27.885 s。evidence 8.805 s，reduction 29.5 ms，均不是主要热点。

### 6.4 GPU

| 项目 | 值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8,188 MiB total；峰值 7,872 MiB（96.14%） |
| utilization | mean 25.50%，max 100% |
| power | max 102.94 W |
| temperature | max 81 °C |
| samples / errors | 1,046 / 0 |

显存余量小；在 4060 Laptop 上不能未经实测提升 worker 并发。低均值利用率主要由 source/I/O/串行等待解释，不能当成模型吞吐承诺。

## 7. SQLite / CAS / digest 审计

6 个 SQLite 均 `integrity_check=ok`、`foreign_key_check=0`：

- `inference-evidence.sqlite3`：八段 lineage 各 51 行，terminal 51；
- `logical-nodes.sqlite3`：logical/processing nodes 各 60；
- artifact registry：20 artifacts、20 locations、51 edges；
- primary completion：1 primary run，completion/outbox/delivery 全 0；
- inference barrier：51 part completions 全 `SUCCEEDED`；
- work scheduler：41 windows、205 items、205 attempts、41 pending deliveries。

Raw provider CAS 有 4 个物理 blob、46 bytes，总体 SHA mismatch 为 0；51 条 immutable evidence 通过内容去重引用这些 blob。

| artifact | SHA-256 |
|---|---|
| source MCAP | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| report | `61914c131f04286b0d1fab1979aeac3b9756847dd0ee73bcffcd8ce3b9487717` |
| trace | `6bc90de0cfad35c27f6329004f92a49ee40ea33c1359dc13751c10c4e990e6da` |
| participation | `7ec9fe332a33d5cc199fc57d9e7a73f2f76a2562956630f6d378a68670095654` |
| GPU telemetry | `d434466ce284eb7fdfdd55c53bb483ceab592697603ded6b8e4669b376db6691` |
| error sidecar | `b176e542b9fe777f0d01e2ece8f880bd27a4af0d92b36876e0263585a784b121` |

交叉关系均成立：report 中记录的 trace/participation/GPU/source digest 与实际重哈希一致，`participation.runtime_fragment_sha256` 等于 trace SHA，unclassified span 为 0。

## 8. 运行序列说明

- **r8**：旧 hardening 版本的历史 baseline；`INCOMPLETE`，provider sum 100.466 s，source 较热。
- **r9**：endpoint preflight 正确捕获 `model_version=local` 与 binding 要求 `local-2026-08-06` 的漂移，未进入 canonical run。
- **r10**：因 endpoint 状态/端口冲突导致运行超时，不作为 benchmark。
- **r11**：验证 durable idempotency replay 和报告复现；coarse 多为旧 key replay，不作为 fresh latency。
- **r12**：独立 endpoint state，初始 idempotency 表为空，51 个新 key，唯一用于本文主性能与语义结论。

## 9. 生产准备度

### 已证明

- 完整 40.89 秒 MCAP admission 和六路 quality ledger；
- 单 worker 本地真实 Qwen QA（51/51 成功）；
- checkpoint manifest、HF revision、parser contract 和 model binding 一致；
- raw→parsed→selected→enriched 八段 lineage、CAS、SQLite、trace、参与矩阵和 GPU telemetry；
- cold source export 与模型推理热点的端到端量化；
- `DEGRADED` 质量门 fail-closed；
- durable idempotency 的跨请求/独立状态恢复行为。

### 尚未证明/仍未闭合

- 真实 Qwen event/action/boundary/fusion 和 committed publication；
- 真实 Qwen stream adapter（当前 fixture-only）；
- quality-gate failure 的 primary completion/outbox/delivery/finalization 终态；
- approved mapping、PostgreSQL/Supabase canonical stores、R2/RunPod 外部参与；
- Mage/Qwen 金丝雀、影子路由、并发对照及生产容量；
- 全成功路径的 production receipt 和 authority。

## 10. 下一步

1. 把 `source.media.package_binding` 的 cold `EXPORT_TIME`（746.188 s）作为第一优先级，建立可观测的导出缓存/对象存储复用并分别定义 cold/warm SLO。
2. 逐坐标分析四个 `DEGRADED` 的 frame、prompt、源质量和模型输出；不能通过放宽 parser 或改写状态绕过质量门。
3. 接入真实 stream driver，沿用同一 trace、lineage、CAS 和 participation contract，再跑全阶段。
4. 为 `INCOMPLETE` 建立明确终态 completion/outbox/finalization，避免 primary run 长期 `RUNNING`。
5. 接入批准 mapping 与 PostgreSQL/Supabase/R2/RunPod 后，以相同 source hash、checkpoint manifest、窗口计划和 telemetry 重测；r12 只能作为本地 conformance baseline。

## 11. 最终判定

```text
真实 Qwen QA：已验证（51/51 terminal succeeded）
source cold 导出：已量化，第一热点约 854.232 s union
单 worker inference：已量化，第二热点约 276.373 s union
完整真实模型全阶段：未验证
stream：FIXTURE_ONLY
publication/outbox/finalization：未闭合
质量门：正确 fail-closed
生产就绪：否
最终状态：INCOMPLETE / RUN_NOT_COMPLETABLE
```

本报告只描述隔离 worktree 的本地运行；未触碰主 worktree 的 `/web`。实现已分批提交并推送到 `codex/local-qwen-production-e2e-20260806`，draft PR 为 `#4`，不会自动合并。
## 12. 静态检查与测试

- 聚焦回归（本轮全套 canonical/Qwen 相关测试）：`108 passed in 367.53s`；mypy 修复后的 endpoint/adapter/real-model/binding/report 子集再次 `55 passed in 82.02s`；locked uv CI 环境中的 endpoint + binding 测试为 `19 passed in 34.50s`。
- 全量回归：`1876 passed, 20 skipped in 1999.86s`；仅 pytest-asyncio fixture loop-scope deprecation warning。
- Ruff check：20 个本轮 Python 文件全部通过。
- Ruff format check：20 个文件均已格式化。
- `py_compile`：使用临时 `PYTHONPYCACHEPREFIX` 后通过。
- `git diff --check`：通过；仅 CRLF→LF 常规提示。
- FastAPI health route 已改为显式注册，消除不同 mypy 版本对动态 decorator 的 `misc` / `untyped-decorator` 分歧。
- Locked CI 环境执行 `uv sync --locked --dev` 后，`uv run mypy` 为 `Success: no issues found in 278 source files`。全局 Anaconda 环境直接运行 mypy 会因缺少 locked stubs/optional dependency typing 产生环境差异，仓库和 CI 判定以 `uv.lock` 环境为准。