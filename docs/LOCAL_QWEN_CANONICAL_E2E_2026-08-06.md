> **历史说明（2026-08-07 更新）**：本文件保留 r8 的历史基线。当前硬化代码和主审计请以 `docs/LOCAL_QWEN_CANONICAL_E2E_R12_2026-08-07.md` 及 r12 artifact 目录为准；其中 checkpoint identity、durable idempotency、严格 parser 和 r12 冷启动热点数据已更新。

# Qwen3-VL-4B 本地 Canonical 全链路与热点审计报告

- **审计日期**：2026-08-06
- **代码隔离 worktree**：`D:\tmp\robata-local-qwen-e2e-20260806`
- **运行状态目录**：`D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r8-20260806`
- **审计输出目录**：`D:\tmp\robata-qwen-run-20260806\reports\canonical-qwen-full-r8-20260806`
- **报告格式**：`robata-local-qwen-canonical-e2e-v2`
- **证据等级**：`LOCAL_CONFORMANCE`
- **生产资格**：`false`
- **Canonical authority**：`false`

> **最终状态：`INCOMPLETE / RUN_NOT_COMPLETABLE`**
>
> 本报告记录一次使用完整本地 MCAP 源、单 worker 本地 Qwen3-VL-4B-Instruct、canonical source、调度、QA、证据、存储审计和运行时遥测的端到端执行。它证明本地真实 Qwen QA 路径及其证据链可以被审计；它**不**证明完整生产链路已经成功，也不构成生产就绪或 canonical authority 声明。

## 1. 执行结论

本次运行的失败原因明确且受控：`QA_DENSE` 对六路必需相机坐标的最终归约中，`cam_03`、`cam_05`、`cam_06` 仍为 `DEGRADED`。质量门按 fail-closed 规则阻止后续完成，因此运行被标记为 `INCOMPLETE`，错误码为 `RUN_NOT_COMPLETABLE`。

以下事实已被实际运行和审计证据证明：

- 本地 Qwen endpoint 已通过 `READY` 健康检查；
- `QA_COARSE` 与 `QA_DENSE` 合计 **51 次真实 Qwen terminal 调用**均为 `SUCCEEDED`，且输出均有效；
- raw response、解析后的 claim、选择结果与 enriched output 的八段 lineage 均为 51 行，`lineage_complete = true`；
- scheduler 的 205 个 work item 与 205 个 attempt 均为 `SUCCEEDED`；
- SQLite 完整性和外键检查、CAS 文件散列检查均通过。

但这些成功不能覆盖被质量门阻断之后的真实模型下游阶段。真实 Qwen 的 `EVENT_PROPOSAL`、`ACTION_EVIDENCE`、`BOUNDARY_REFINEMENT`、fusion，以及最终 publication 都没有在本次运行中闭合。流式 route 仍是 fixture-only，不能作为真实 Qwen stream inference 的证据。

`report.json` 中的 `ok` 为 `false`，`receipt` 为 `null`；不得把本次结果表述为完成的 canonical receipt、生产成功或发布成功。

## 2. 运行范围与模型绑定

| 项目 | 值 |
|---|---|
| 模型 | `Qwen3-VL-4B-Instruct` |
| 模型版本 | `local-2026-08-06` |
| provider | `local-huggingface` |
| adapter | `local-hf-loopback-adapter-v1` |
| endpoint | `http://127.0.0.1:8101` |
| endpoint 预检 | `READY`，`loaded = true` |
| endpoint 并发度 | `1` |
| Robata 执行模式 | `single-worker-production-policy-local-conformance` |
| worker 并发度 | `1` |
| source profile 选项 | `allow_unapproved_profile = true` |
| 生产资格和 authority | 均为 `false` |

本地 adapter 的配置与模型绑定在代码中显式固定为非生产、非 authoritative。单 worker、单 call-part、单 batch 的边界是本次测量条件的一部分，不能据此推导并行生产能力。

### 2.1 Mapping approval 边界

本次 mapping 使用 `config/genrobot-observed-v0.json`。该配置明确记录 `approval_status = UNAPPROVED` 且 `approved = false`。运行时传入 `--allow-unapproved-profile`，报告中的 `allow_unapproved_profile = true` 正是这项显式开发授权的审计投影。

这只允许本地 conformance 执行读取该 development profile，**不**会把 mapping 变为 `APPROVED`，也不改变 `LOCAL_CONFORMANCE`、`production_eligible = false` 或 `canonical_authority = false`。

### 2.2 Canonical 逻辑时钟与实际 trace 时间

canonical composition 使用固定逻辑时钟 `2026-07-20T00:00:00Z`。它用于可复现的 canonical 语义、identity 和持久化时间字段，不能被误读为本次机器执行的实际时间。

本次实际审计发生在 2026-08-06：`participation.json` 的 `observed_at` 为 `2026-08-06T21:59:07.754299Z`，trace 的角色为 `CONTROL`，时钟域为 `PROCESS_LOCAL_MONOTONIC`。性能和 span 数据必须使用实际 trace 的 monotonic offsets 与该观测时间解释，不能用固定逻辑时钟代替。

### 2.3 模型 payload 可复现性边界

本次报告可审计的模型身份是 provider、`Qwen3-VL-4B-Instruct`、`local-2026-08-06`、adapter 版本和 endpoint health。r8 artifact 中**尚未**把 checkpoint 身份、Hugging Face revision、weights、tokenizer 和 processor 的 manifest 作为不可变 artifact 绑定到此次运行。

因此，endpoint 已加载且 51 次 Qwen 调用有完整 response lineage，并不等于模型 payload 已获得 checkpoint 级可复现性证明。该缺口必须在未来运行中以版本、digest 与 artifact binding 闭合。

## 3. 完整输入源与 source admission

本次不是裁剪后的小样本执行。完整文件 `sample-medium.mcap` 已被读取、检查并准入；`semantic_stage_outcomes.source.status` 为 `ADMITTED`，且 media-quality report 为完整请求区间持久化的依据。

| 项目 | 值 |
|---|---:|
| 源文件 | `D:\Github\Robata\data\source\sample-medium.mcap` |
| 文件大小 | 130,303,923 bytes |
| SHA-256 | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| MCAP header library | `libmcap` |
| MCAP header profile | `Genrobot` |
| MCAP messages | 16,210 |
| channels | 17 |
| 相机 channels | 6 |
| 首条消息时间 | `1781051907238265000 ns` |
| 末条消息时间 | `1781051948128720000 ns` |
| 原始 MCAP 消息时间跨度 | 40.890455 s |
| requested maximum duration | 41.000000000 s |
| admitted interval | `[0, 40.833513001 s)` |
| admitted recording duration | 40.833513001 s |
| `window_limited` | `false` |
| media-quality semantic SHA-256 | `b3d081f9b396fa17dd1d405b6e9d8317307cd850f54c29d0db562e2b68ba186f` |

六路相机均为 `foxglove.CompressedImage`、`protobuf`、`h264`，各自原始消息数为 1,226，时间戳单调。admission 后，每路都有 1,225 个 timing frame 和 82 个 media-quality observation：

| 相机 | admitted timing frames | media-quality observations |
|---|---:|---:|
| `cam_01` | 1,225 | 82 |
| `cam_02` | 1,225 | 82 |
| `cam_03` | 1,225 | 82 |
| `cam_04` | 1,225 | 82 |
| `cam_05` | 1,225 | 82 |
| `cam_06` | 1,225 | 82 |

需要区分三个时间和计数概念：40.890455 s 是原始 MCAP 的全消息跨度；40.833513001 s 是六路相机的 admitted recording duration；82 是每路质量观测采样数，不是完整帧数。`window_limited = false` 表示本次没有因请求窗口截断可用录制区间。

### 3.1 Admitted timing frame 与 Qwen image input 的差别

六路 admitted timing frame 合计为 **7,350**，即 6 × 1,225。它们不是对模型逐帧投喂的计数。trace 的 inference counters 显示，本次真实 Qwen provider 实际接收 **306 个 image inputs**：`QA_COARSE` 为 246，`QA_DENSE` 为 60。跨两个 QA stage 的 `inference.unique_images` 为 300。

因此，本次 51 次调用证明的是对选定图像输入的真实 Qwen QA，而不是对 7,350 个 admitted timing frame 逐一进行模型推理。完整 source admission、帧选择和模型 image input 是三个不同的审计层级。

## 4. 参与边界与语义阶段结果

`participation.json` 覆盖了全部七个必需边界，所有边界都具有 `MEASURED` 运行时观测；`unclassified_span_count = 0`。整体 coverage 为 `FAILED`，唯一声明失败的边界是 `REDUCTION`，原因是 dense QA 未解析出全部六路必需观测。

| 边界 | 观测 span 数 | participation 状态 | 对应语义结果 |
|---|---:|---|---|
| `ORCHESTRATION` | 4 | `PARTICIPATING` | 运行到质量门终止点 |
| `SOURCE` | 22 | `PARTICIPATING` | `ADMITTED` |
| `SCHEDULING` | 11,636 | `PARTICIPATING` | `SUCCEEDED` |
| `INFERENCE` | 158 | `PARTICIPATING` | `SUCCEEDED` |
| `EVIDENCE` | 1,268 | `PARTICIPATING` | `SUCCEEDED` |
| `REDUCTION` | 2 | `FAILED` | `FAILED_QUALITY_GATE` |
| `PUBLICATION` | 1 | `PARTICIPATING` | `NOT_COMMITTED` |

`PUBLICATION` 被测到只表示运行时看到 publication 边界，例如 completion 开始动作；它不表示 primary completion、outbox 或实际发布已经提交。审计数据明确显示这些项目均未闭合。

## 5. 真实 Qwen QA 调用与证据链

本节的计数来自 `inference-evidence.sqlite3`、`report.json.inference_audit` 和 terminal group。它们是本次真实本地 Qwen QA 调用的可审计证据，而不是 stream fixture 计数。

### 5.1 调用结果

| Stage | terminal 数 | provider | terminal 状态 | 有效输出 |
|---|---:|---|---|---:|
| `QA_COARSE` | 41 | `local-huggingface` | `SUCCEEDED` | 41 / 41 |
| `QA_DENSE` | 10 | `local-huggingface` | `SUCCEEDED` | 10 / 10 |
| **合计** | **51** |  | **全部成功** | **51 / 51** |

所有 terminal 都绑定到 `Qwen3-VL-4B-Instruct` 和版本 `local-2026-08-06`。因此可以确认本地真实模型已在 offline canonical QA 路径中完成 51 次有效调用。

51 个 response 中有 **4 个裸 `GOOD` 标签**。它们不是宽松地猜测为有效 JSON，而是按 `exact-allowed-label-v1` 的严格 bare-label recovery 规则接受。该规则只允许精确的已允许标签；任何额外文本、未知标签或不满足严格解析条件的内容仍应 fail closed。

### 5.2 八段 lineage

| 证据表 | 行数 |
|---|---:|
| `inference_intents` | 51 |
| `model_inference_terminals` | 51 |
| `raw_provider_artifacts` | 51 |
| `raw_provider_responses` | 51 |
| `parsed_provider_claims` | 51 |
| `inference_attempt_selections` | 51 |
| `selected_attempt_outputs` | 51 |
| `enriched_provider_outputs` | 51 |

八张表的计数均等于 terminal 数，`lineage_complete = true`。这条链覆盖 raw provider bytes 到解析、选择和 enriched 输出的完整可追溯关系。

### 5.3 Provider latency

| Stage | 数量 | 最小值 | P50 | P95 | 最大值 | 合计 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `QA_COARSE` | 41 | 1,730 ms | 1,946 ms | 2,099 ms | 2,289 ms | 80,090 ms | 1,953.415 ms |
| `QA_DENSE` | 10 | 1,898 ms | 1,938 ms | 2,243 ms | 2,243 ms | 20,376 ms | 2,037.600 ms |
| **合计** | **51** |  |  |  |  | **100,466 ms** |  |

### 5.4 QA 观测分布

| Stage | `GOOD` | `DEGRADED` |
|---|---:|---:|
| `QA_COARSE` | 44 | 2 |
| `QA_DENSE` | 11 | 3 |

`QA_DENSE` 的 10 个 provider call part 可覆盖重复的 `(package_ordinal, camera_id)` 坐标，因此 dense 原始观测总数为 14，而不是只按 call 数计算。

## 6. QA_DENSE 归约、质量门与 fail-closed 行为

本地 adapter 明确选择版本化策略 `severity-worst-then-earliest-v1`。其含义是：同一 dense 坐标若出现重复观测，先选择更保守的状态；状态相同才选择更早的证据区间。状态严重度顺序为：

```text
GOOD < DEGRADED < UNKNOWN < UNUSABLE
```

本次最终归约如下：

| 坐标 | 输入 observations | 归约结果 |
|---|---|---|
| `0:cam_01` | `GOOD`, `GOOD` | `GOOD` |
| `0:cam_02` | `GOOD`, `GOOD`, `GOOD` | `GOOD` |
| `0:cam_03` | `DEGRADED`, `GOOD` | **`DEGRADED`** |
| `0:cam_04` | `GOOD`, `GOOD` | `GOOD` |
| `0:cam_05` | `DEGRADED`, `GOOD`, `GOOD` | **`DEGRADED`** |
| `0:cam_06` | `DEGRADED`, `GOOD` | **`DEGRADED`** |

未解决的必需坐标恰好是：

```text
0:cam_03
0:cam_05
0:cam_06
```

质量门要求所有六路必需 observation 都解析为 `GOOD`。因此三个 `DEGRADED` 结果必须阻断运行。系统没有丢弃保守证据，也没有将 `DEGRADED` 重写成 `GOOD`；这正是本次 `FAILED_QUALITY_GATE` 和 `RUN_NOT_COMPLETABLE` 的直接原因，也是 fail-closed 行为正确生效的证据。

## 7. Scheduler、stream route 与未闭合的 publication

### 7.1 Scheduler 账本

| 项目 | 值 |
|---|---:|
| expected windows | 41 |
| work items | 205 |
| work attempts | 205 |
| 成功 work items | 205 |
| 成功 attempts | 205 |
| stream window results | 41 |
| stream evidence commits | 41 |

按 stage 的 work item 记录均为 41 个 `SUCCEEDED`：

```text
QA_COARSE_PLAN      41
QWEN_QA_COARSE      41
QWEN_QA_DENSE       41
QWEN_EVENT_PROPOSAL 41
QA_AGGREGATE        41
```

这些 scheduler 状态说明工作计划、依赖、持久化和调度执行参与成功；它们不能单独作为真实 Qwen 下游模型调用的证据。

### 7.2 Stream route 是 fixture-only

对 stream artifact 进行 provider 级审计后的结果为：

| 项目 | 数量或结论 |
|---|---:|
| fixture inference-intent artifacts | 41 |
| fixture accepted-call artifacts | 41 |
| real Qwen stream artifacts | 0 |
| stream window results | 41 |
| route verdict | `FIXTURE_ONLY` |
| fixture driver | `local-conformance-window-mock-v1` |

因此，stream 调度和持久化路径确实参加了运行，但其 inference driver 仍是 fixture。真实 Qwen 证据仅来自前述 offline canonical QA 路径。特别是，名为 `QWEN_EVENT_PROPOSAL` 的 scheduler work item 成功，不得被误读为已经取得真实 Qwen event-proposal 证据。

### 7.3 Outbox、finalization 与 publication 未闭合

| 项目 | 值 |
|---|---:|
| stream delivery outbox | 41 |
| stream outbox delivery 状态 | 41 个 `PENDING` |
| recording finalizations | 0 |
| primary runs | 1 个 `RUNNING` |
| primary completions | 0 |
| detailed results | 0 |
| primary outbox | 0 |
| primary outbox deliveries | 0 |
| action event publications | 0 |
| publication 语义状态 | `NOT_COMMITTED` |

质量门失败后，primary completion、outbox、delivery 和 recording finalization 都没有形成闭环。**一个 `PUBLICATION` span 被测到，不等于存在 committed publication。** 只有 primary completion、primary outbox 和相应 delivery 的持久化提交才能构成已发布事实。

## 8. 运行时间、吞吐量与热点

### 8.1 整体资源数据

| 项目 | 值 |
|---|---:|
| 总 elapsed wall time | 211.9759674 s |
| admitted video duration | 40.833513001 s |
| 相对实时速度 | 5.191 倍慢于实时 |
| 单 worker 视频吞吐量 | 0.1926 video-hour / wall-hour |
| 按本次局部路径外推的 24 小时吞吐量 | 约 4.62 video-hours / day |
| client process CPU time | 141.515625 s |
| client RSS | 599.78 MiB |
| client read I/O | 3.584 GiB |
| client write I/O | 249.25 MiB |

上述吞吐量仅适用于已经实际执行到质量门的路径：完整 source、scheduler、真实 Qwen QA、evidence 和 quality gate。它不包含真实模型的下游 event、action、boundary、fusion 或 committed publication。因此它**不是**完整生产链路吞吐量，也不得用作生产容量承诺。

### 8.2 Trace stage wall-time union

| Stage | span 数 | wall-time union | 占总 elapsed 的比例 |
|---|---:|---:|---:|
| `ORCHESTRATION` | 4 | 208.271 s | 98.25% |
| `SOURCE` | 22 | 87.227 s | 41.15% |
| `SCHEDULING` | 11,636 | 7.035 s | 3.32% |
| `INFERENCE` | 158 | 118.813 s | 56.05% |
| `EVIDENCE` | 1,268 | 9.265 s | 4.37% |
| `REDUCTION` | 2 | 0.0254 s | 0.012% |
| `PUBLICATION` | 1 | 0.0046 s | 0.002% |

这些 union 存在嵌套，不能相加。`ORCHESTRATION` 接近整个外层执行范围。就可观察的热点而言，`INFERENCE` 为 118.813 s，`SOURCE` 为 87.227 s；provider 的 51 次 latency 合计为 100.466 s，约占总 elapsed 的 47.39%。`INFERENCE` 的 stage union 比 provider latency 多约 18.347 s，包含请求准备、序列化、HTTP、materialization、lineage 等外围工作。

`REDUCTION` 的计算时间本身不是性能瓶颈；它阻断流程的原因是质量结论，而不是归约耗时。

## 9. GPU telemetry

GPU telemetry 的测量状态为 `MEASURED`，每 1,000 ms 采样一次，共 195 个样本，`errors` 为空。

| 项目 | 值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU index | 0 |
| VRAM 总量 | 8,188 MiB |
| VRAM 峰值 | 7,877 MiB |
| VRAM 峰值占比 | 96.20% |
| GPU 利用率均值 | 45.487% |
| GPU 利用率峰值 | 100% |
| 功耗峰值 | 107.39 W |
| 温度峰值 | 80 °C |

当前单 worker 配置在显存峰值时仅剩很小余量。峰值 100% 说明运行中曾充分使用 GPU，但 45.487% 的均值不能被解读为模型 kernel 始终受限；完整 wall time 还包括 source、I/O、请求准备、串行策略和证据持久化。是否能安全提升并发或部署到另一种硬件，必须在相同完整负载和 telemetry 下另行验证。

## 10. SQLite 与 CAS 审计

### 10.1 SQLite

审计发现 6 个实际 SQLite 数据库，排除了 `-wal` 与 `-shm` 临时文件。所有数据库均满足：

```text
PRAGMA integrity_check = ok
PRAGMA foreign_key_check = 0 violations
```

`sqlite_database_count = 6`，`sqlite_integrity_failures = 0`。

| 数据库 | 大小 | 关键审计事实 |
|---|---:|---|
| `inference-evidence.sqlite3` | 19,333,120 bytes | 八段 inference lineage 各 51 行，51 个 terminal 均为 `SUCCEEDED` |
| `logical-nodes/logical-nodes.sqlite3` | 229,376 bytes | `logical_nodes` 与 `processing_run_nodes` 各 60 行 |
| `mcap/.../artifact-registry/registry.sqlite3` | 131,072 bytes | 20 artifacts、20 artifact locations、51 artifact edges |
| `primary-completion.sqlite3` | 147,456 bytes | 1 个 primary run，completion、outbox、delivery 均为 0 |
| `runs/.../inference-call-barrier.sqlite3` | 417,792 bytes | 51 个 part completion 均为 `SUCCEEDED`，2 个 reduction 记录 |
| `work-scheduler.sqlite3` | 3,342,336 bytes | 41 expected windows、205 work items、205 attempts、41 个 pending stream delivery |

SQLite 审计通过表示记录没有观察到完整性或外键损坏；它不改变 primary completion 仍未提交这一事实。

### 10.2 Raw provider CAS

| 项目 | 值 |
|---|---:|
| `raw_provider_artifacts` 行数 | 51 |
| `raw_provider_responses` 行数 | 51 |
| 物理 CAS 文件数 | 4 |
| 物理 CAS 总字节数 | 46 bytes |
| CAS SHA-256 mismatch | 0 |

CAS 审计会将以 SHA-256 命名的物理文件重新哈希并与文件名比对。本次没有 mismatch。物理文件数少于 51 是内容去重的表现，不是 51 条 immutable evidence 记录缺失；这 51 条记录仍可在 inference evidence 账本中审计。

## 11. 审计 artifact 与摘要 digest

| Artifact | 路径 | SHA-256 |
|---|---|---|
| source MCAP | `D:\Github\Robata\data\source\sample-medium.mcap` | `9fd5094bf29cd4ee50cd8c7d8c053e89d1c93660a0f4e57daaa726bae2b6156c` |
| report | `...\reports\canonical-qwen-full-r8-20260806\report.json` | `3d8c1278864a7f2d065a9c31fddb80ddefeb19ee085e07dae4b107446b64af6a` |
| trace | `...\reports\canonical-qwen-full-r8-20260806\trace.json` | `4bda1e144b5e446fc9644b109caceac1de6b168ca9b56728730f213822cce017` |
| participation | `...\reports\canonical-qwen-full-r8-20260806\participation.json` | `8d4dc4789fe5df3e871b3489766ba949008e4ff2848c55e02f1e6af627b21528` |
| GPU telemetry | `...\reports\canonical-qwen-full-r8-20260806\gpu-telemetry.json` | `c9662fb68c7689ba05d7b0c787aece977effedf0ddf6135cc3101bbe319f127f` |
| error sidecar | `...\reports\canonical-qwen-full-r8-20260806\error.json` | `bbf0ad1553380f9750328c7e2539fca74ff50a8d114029c1fa885a70af154483` |

交叉校验结果：

```text
trace 文件 digest = report.runtime.trace_sha256
participation 文件 digest = report.runtime.participation_sha256
participation.runtime_fragment_sha256 = trace 文件 digest
GPU telemetry 文件 digest = report.runtime.gpu_telemetry_sha256
source 文件 digest = report.source.sha256
所有 SQLite integrity 和 foreign-key 检查通过
所有 CAS 文件名散列检查通过
```

`participation.json` 的 trace ID 为 `3a5c302c-0041-482d-a415-d88849efa6ab`，观测时间为 `2026-08-06T21:59:07.754299Z`。

## 12. 相关实现与测试记录

本次运行由 `scripts/run_local_qwen_canonical_e2e.py` 生成 sidecar、trace、participation、GPU telemetry 和统一报告。相关实现包括本地真实模型绑定、严格 loopback adapter、QA_DENSE 重复坐标归约、canonical composition、runner、MCAP source 与 frame/media cache 路径。

### r8 收尾回归记录

本报告保留的 r8 收尾回归结果为：

```text
141 passed in 624.06s
```

该数字是本次 r8 运行的历史测试记录。审计 sidecar 本身不保存当时的完整 pytest 命令行，因此不能把本文档修复后的任意本地重跑结果替代为该历史结果。

相关回归覆盖的模块类别包括：

- local HF adapter、真实模型和 canonical model binding；
- QA_DENSE duplicate-coordinate reduction；
- E2E trace 与 participation manifest；
- MCAP source preparation、canonical offline pipeline 与 canonical local command；
- bounded media、media quality、layered media cache 及其并发行为。

补充记录：

```text
QA_DENSE duplicate-coordinate reduction: 4 passed
local real-model 和 binding 复跑: 10 passed
Ruff format/check: passed (16 files)
py_compile: passed
failure-path wrapper smoke: passed
sidecar、source、SQLite、CAS 独立复核: passed
```

重复坐标归约测试覆盖更保守状态优先、同状态时更早 interval 优先，以及输入顺序不改变结果。测试结果验证 fail-closed 归约逻辑，而不是通过放宽质量门来取得成功。

## 13. 生产准备度与下一步

### 已经得到本次证据支持的内容

- 完整 MCAP source 的读取、检查、准入和六路相机 materialization；
- 单 worker 本地真实 Qwen QA 调用及 51 次有效 terminal；
- raw bytes 到 enriched output 的完整 QA lineage；
- scheduler、work ledger、trace、participation、CPU、I/O 与 GPU telemetry；
- SQLite 与 CAS 的可审计完整性；
- 当必需相机质量不合格时正确 fail-closed。

### 本次没有证明的内容

- 真实 Qwen 的 `EVENT_PROPOSAL`、`ACTION_EVIDENCE` 与 `BOUNDARY_REFINEMENT`；
- 完整 real-model fusion、committed detailed result 和最终 publication；
- primary completion、primary outbox、delivery、recording finalization 的终态闭环；
- 真实模型 stream adapter。当前 stream route 仍是 `local-conformance-window-mock-v1` fixture；
- checkpoint、Hugging Face revision、weights、tokenizer 和 processor manifest 的不可变 artifact binding；
- 外部生产基础设施或生产授权，以及完整成功路径上的生产吞吐量和并发容量。

### 应优先处理的事项

1. 分析 `cam_03`、`cam_05`、`cam_06` 为何产生 `DEGRADED`，从 frame selection、prompt、输出约束或源质量中定位原因。不得通过把 `DEGRADED` 改写为 `GOOD` 来绕过质量门。
2. 用明确的真实模型 adapter 替换 stream fixture，并把实际 provider artifacts 纳入相同的 evidence 审计。
3. 为 `INCOMPLETE` 结果建立明确的终态 completion、outbox 与 finalization 语义，避免 primary run 长期保持 `RUNNING`。
4. 在上述缺口闭合后，以同一 source hash、窗口计划、模型配置和完整 telemetry 重跑，再评估性能和发布条件。

## 14. 最终判定

本次本地 Qwen QA 路径已从接口可达性验证推进到可复核的端到端证据：完整 source 已准入，51 次真实 Qwen QA 调用成功，lineage、scheduler、SQLite、CAS、trace 和 GPU 数据均可审计。

但质量门对 `cam_03`、`cam_05`、`cam_06` 的 `DEGRADED` 结论进行了正确阻断；真实 stream inference 仍未接入；publication、outbox 与 finalization 未闭合。因此最终结论只能是：

```text
真实 Qwen QA 路径：已验证
完整真实模型全阶段链路：未验证
stream 路由：调度和持久化已参与，但 inference 为 fixture
publication、outbox、finalization：未闭合
质量门：正确 fail-closed
生产就绪：否
最终运行状态：INCOMPLETE / RUN_NOT_COMPLETABLE
```
