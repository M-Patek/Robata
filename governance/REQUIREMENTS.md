# Robata 生产环境需求参考（迁移稿，非权威指导）

> **文档身份**：本文件从 `archive/old_mvp/REQUIREMENTS.md` 迁移而来，记录生产环境目标、容量假设和未来产品设计。它是当前 `governance/` 的导航参考，不是产品契约、审批流程、任务服务、实现状态报告或生产资格证明。
>
> **来源**：`archive/old_mvp/REQUIREMENTS.md`（原始日期 2026-07-19，原文标注“尚未实现”）。历史文件保留用于追溯；本迁移稿不改变其内容含义。
>
> **权威性边界**：
> 1. wire contract、identity/hash、logical key、idempotency key、fence 和语义投影：以 `schemas/`、`schemas/schema-catalog.json` 及注册 schema 流程为准。
> 2. 可执行行为：以 tracked source、tests 和 conformance fixtures 为准。
> 3. 当前施工路线：以 `governance/BLUEPRINT.md` 和相关 module card 为准。
> 4. 本文件中的生产目标、SLA、容量、成本和云服务选型均为需求参考或待验证假设，不能单独宣称“已支持”。
>
> **当前资格状态（2026-07-26）**：本地 contract/conformance、unit 和 integration 回归已通过；真实 NVDEC、两张 H100/vLLM、生产 R2/broker/object store、代表性标签及 24 小时 soak 尚未测量，生产资格仍为 `NOT_MEASURED`。

## 标签说明

- **PRODUCTION-REQUIREMENT**：生产环境希望达到的行为或能力目标；不是当前实现承诺。
- **PRODUCTION-SLA-TARGET**：生产 SLA/吞吐目标；必须由真实环境测量后才能转为 `QUALIFIED`。
- **PRODUCTION-ARCHITECTURE-TARGET**：目标部署分工；云服务并不自动提供缺失的 CPU/GPU 计算能力。
- **PRODUCTION-CAPACITY-HYPOTHESIS**：容量估算或前提，允许被实测推翻。
- **EXTERNAL-NOT-MEASURED**：依赖真实硬件、provider、数据或生产服务，当前未验证。
- **REFERENCE / BACKGROUND**：设计背景、词汇或历史材料，不覆盖权威契约。
- **PROPOSED / ESTIMATE**：未来方案或粗略估算，不是批准后的范围或预算。

## 当前实现与生产目标的分界

| 目标 | 本地状态 | 生产结论 |
| --- | --- | --- |
| 21 类 QA、证据链、恢复、幂等、outbox | 已有本地实现和回归证据 | 生产硬件、provider、代表性数据仍未资格验证 |
| AI 预标注（7B/vLLM） | 接口/协议基础存在，完整生产链未验证 | `NOT_MEASURED` |
| 结构化/embedding 视频搜索 | 设计目标与部分本地基础 | 非生产服务证明 |
| R2 + CPU/NVMe worker + H100 | 架构方向可行 | R2 仅对象存储；worker 规格、带宽和扩展效率未实测 |
| 500 recording-hours/day、T+1/T+3 | 需求目标 | 当前单进程本地基线不是容量证明 |

---

## 原始需求正文（迁移保留）


面向生产的需求与设计(Kev, 2026-07-19)。**尚未实现** —— 本文是三条工作线的需求基线和容量评估。
附录 A(21 类问题清单)、附录 B(片段处理与后处理设计)附于文末,本文自洽独立。

## 0. 概述

> **本节状态：PRODUCTION-REQUIREMENT / 目标范围，不证明当前实现或资格。**


三条工作线,共用同一批 GPU 与抽帧产物:

1. **AI QA** —— 逐 clip 打标(起止时间 + 问题),视频级判 pass/warning/fail。
2. **AI 预标注** —— 非 fail 视频全量进入,按 annotation principal 生成动作分段草稿。
3. **AI 视频搜索** —— 自然语言检索,返回具体 clip(start→end),点击跳播。

生产量级:**500 小时/天,每天新增 500 小时**。

---

## 1. Quality Assessment

> **本节状态：PRODUCTION-REQUIREMENT / 合同候选；结果形状和判定语义以发布 schema、源码、测试和 conformance 为准。**


### 1.1 数据模型变更

> **本节状态：PRODUCTION-REQUIREMENT / 数据模型目标；不单独授权 wire-shape 或 schema 变更。**


QA 结果从 **`{result, issue, severity, description}` per video** 改为 **逐 clip 打标**:

```
clip_mark: { start_sec, end_sec, issue, confidence }
```

- 一个视频可有**多段**问题 clip,需**全部标出**。
- `confidence` = 该问题判定的置信度(机检出的为检测器置信度)。
- (相较旧模型:去掉 `severity` / `description`,新增 `confidence` 与逐段时间。)

### 1.2 视频级判定:pass / warning / fail

> **本节状态：PRODUCTION-REQUIREMENT / 判定目标；当前行为请查源码和 conformance。**


| 判定 | 条件 | 处置 |
|---|---|---|
| **pass** | 全程无问题 | 进入标注 |
| **warning** | 有**局部**问题 clip(其余可用) | **进入标注**,坏段随标记带入下游 |
| **fail** | **全程**性问题(全程黑屏、全程设备戴反等) | **删除**(唯一需删除的一档) |

**warning ≠ 删除。** 只有 **fail** 才删。warning 视频保留,其问题段以 `clip_mark` 形式标记,
下游(标注 / SLAM / 骨骼)据此决定跳过或降权 —— 详见附录 B。

### 1.3 问题 → 判定 的映射

> **本节状态：PRODUCTION-REQUIREMENT / 映射目标；21 类词汇的执行来源是仓库 QA vocabulary 和测试。**


21 类问题按"时间局部 vs 整条属性"决定产出 warning 还是 fail(完整映射与原因见附录 B):

- **时间局部 → warning(打标)**:黑屏\*、画面损坏、移动过快、手不在中心、静止超5秒、头发遮挡、
  局部无关动作、任务无关动作、手臂被遮、双手交叉、过暗过曝\*、无关人/动物入镜
- **整条属性 → fail(删除)**:设备戴反、镜头模糊、任务未完成、真实性欠缺、着装暴露
- **不可判(跨片/任务目录)**:缺乏多样性、做成了别的任务

\* 若整条都黑/都过曝则升为 fail。

### 1.4 SLA

> **本节状态：PRODUCTION-SLA-TARGET / NOT_MEASURED；T+1/T+3 与 500h/day 尚未完成真实生产资格验证。**


- **全量 AI QA 在"当日上传完毕(T)后 1 天内"完成 → T+1。**
- 稳态:每天 500 小时新视频,QA 吞吐须持续跟上 500h/天。

---

## 2. Annotation(AI 预标注)

> **本节状态：PRODUCTION-REQUIREMENT / 目标能力；本地 QA 主链路不等于完整生产预标注服务。**


### 2.1 流程

> **本节状态：PRODUCTION-REQUIREMENT / 目标流程；仍需真实模型、治理标签和人工验收。**


- QA 结束后,**所有非 fail 视频(pass + warning)全量进入**标注流程。
- 按 **annotation principal** 进行 **AI 预标注**(动作分段草稿;人在环里 accept/改)。
- warning 视频的 QA 标记**带入**标注,坏段可据此跳过/降权。

### 2.2 SLA

> **本节状态：PRODUCTION-SLA-TARGET / NOT_MEASURED；不得用 mock 或本地 CPU 基线宣称达标。**


- **全量 AI 预标注在"QA 完毕后 2 天内"完成 → 约 T+3。**
- 稳态:同样是 500h/天。deadline 的宽松只提供缓冲,**不减少稳态吞吐要求**。

---

## 3. AI 视频搜索

> **本节状态：PRODUCTION-REQUIREMENT / 目标能力；作为产品目标记录，不是已交付生产搜索服务。**


**总目标**:搜索栏输入自然语言 → 返回匹配的**具体 clip(视频里的 start→end 段落,非整条视频)**,
点击直接跳播。让全语料里"某个动作-物体"发生在哪些片段可被检索。

### 3.1 已有可复用(标注阶段的产出)

> **本节状态：REFERENCE / 可复用设计；实际可用字段和播放行为由源码、schema、测试决定。**


- **clip 边界** = Assignment 1 的 segment(带 `start_time_sec` / `end_time_sec`)—— "返回具体
  clip 而非整条视频"已由标注解决。
- **action-object pair** = `structured_labels`(verb / noun / attributes / location / hand)。
- **按段播放** = 播放器已有"跳到 start、播到 end"能力。

### 3.2 MVP 阶段(只用 structured_labels,零 GPU)

> **本节状态：PROPOSED / 非承诺 MVP；需要单独立项和实现证据。**


- **做什么**:汇总所有 segment 成 clip 索引(每条 = start/end + verb/noun/attributes);把自由文本
  动词**聚类成规范动作族**(前置);查询解析成 `动词族 + noun` 去过滤,返回 clip 并按段播放。
- **目的**:用标注现成产出做出**精确、可解释、零 GPU** 的 faceted clip 搜索;先跑通"clip 索引 + 跳播"链路。
- **前置**:**动词聚类** —— 现有 `VERB_LEMMAS` 只归时态,不归同义(wipe/scrub/wash);需在其上加一层
  规范动作族映射(48 个动词,人工映射表最可控;或 LLM 辅助 / embedding 聚类)。

### 3.3 Embedding 阶段(语义检索升级)

> **本节状态：PROPOSED / EXTERNAL-NOT-MEASURED；Supabase/pgvector 与视觉 embedding 仍属外部部署与验证项。**


- **做什么**:把每个 clip(先文本=标签,后视觉=抽样帧)与查询嵌成向量,按相似度检索,与 facet 过滤
  **混合**(召回靠向量、精度靠 facet)。
- **目的**:覆盖换句话/近义、以及"像这样的 clip"这类 facet 抓不到的语义匹配。
- **落地**:文本向量放 **Supabase pgvector**(便宜、无 GPU);视觉向量在 **GPU 上离线跑**(更强、更重)。

---

## 4. 基础设施与容量评估

> **本节状态：PRODUCTION-ARCHITECTURE-TARGET / 假设；记录 R2、CPU/NVMe worker、GPU、数据库的目标分工，不是部署承诺。**


### 4.1 架构分工

> **本节状态：PRODUCTION-ARCHITECTURE-TARGET / EXTERNAL-NOT-MEASURED；R2 是存储，不提供 CPU；CPU/NVMe worker 与 H100 服务需单独部署和实测。**


| 组件 | 承载 |
|---|---|
| **Cloudflare R2** | 视频存储 + vendor 流播(egress 免费) |
| **RunPod GPU(vLLM)** | AI QA + 预标注 + 视觉 embedding(离线任务) |
| **Supabase** | 登录、QA/标注结果、pgvector 文本向量 |

- **上传一次**:视频 → R2;R2 → GPU 是自动、免费的拉取,不是二次上传。

### 4.2 抽帧共享(feed once)

> **本节状态：PRODUCTION-PERFORMANCE-REQUIREMENT / 设计目标；必须由真实 I/O、缓存和端到端 profile 证明。**


- **AI QA 与 AI 预标注共用抽帧**。QA 的 Stage 1 本就要解码视频 → 抽帧(≥2 fps)并**缓存**;
  预标注(2 天后)**读缓存帧**,不重拉视频、不重解码。视频只到 GPU 一次。
- 帧缓存约 5–7TB(存 2–3 天),放 R2 约 ~$100/月。

### 4.3 GPU 容量评估:2 块 H100(vLLM)能否达标

> **本节状态：PRODUCTION-CAPACITY-HYPOTHESIS / NOT_MEASURED；仅为估算，不能视为两张 H100 已支持 500h/day。**


一块 H100 = 24 GPU-小时/天;两块 = **48 GPU-小时/天**。粗估日负荷(需实测校准):

| 组合 | 每天 GPU-小时 | 结论 |
|---|---|---|
| AI QA(优化后) | ~2 | 便宜 |
| 预标注 · **7B** · 全量 500h | ~30 | — |
| **QA + 预标注 7B** | **~32** | ✅ **塞进 48,余量 ~33%** |
| 预标注 · **32B** · 全量 | ~100 | ❌ 需 4–5 块 |

**结论:2 块 H100 + 7B 草稿模型 + vLLM,可在 T+1(QA)/ QA+2(预标注)内完成 500h/天全量任务,有 ~33% 余量。32B 跑量则不够。**

**前提(缺一不可):**
1. **换 vLLM** —— 现 runner 用 transformers 直跑,慢 5–10 倍,达不到。
2. **7B 够用** —— 预标注是草稿(人工修),7B 通常够,但须在鱼眼画面上实测。
3. **吞吐实测** —— 上表按 7B/vLLM ~40k token/秒估算,有 ±2 倍不确定;若实测减半则需第 3 块。
4. **数据 I/O** —— 5TB/天入站,需 pod ≥1–2.5 Gbps 且 prefetch(和计算重叠、被藏起来);带宽 <0.5 Gbps 时传输成为瓶颈。

### 4.4 成本

> **本节状态：ESTIMATE / 非承诺；价格、租赁和存储成本随供应商和时间变化。**


| 2 块 H100 24/7 | 每月 |
|---|---|
| 按需(on-demand) | ~$3,600–5,100 |
| 月度预留(reserved,省 25–30%) | ~$2,600–3,600 |

vLLM 开源免费,成本纯为 GPU 租金;帧缓存存储 ~$100/月。**当前已租 1 块,酌情增租至 2 块。**

### 4.5 Token 预算(若按 API 视角)

> **本节状态：ESTIMATE / 非承诺；需要真实模型、token、带宽与 provider 账单校准。**


- **AI QA**:优化后 ~$0.01–0.03/小时视频 → 建议上限 **~$0.03/小时**(预计实际落 $100–300/月)。
- **AI 预标注**:密集得多,**~$0.30–0.50/小时视频(自建 7B)**,约为 QA 的 10–15 倍。
- 此量级下**预标注必须自建(pod)**,按 GPU 卡队 size,不按 token 账单。

---

## 5. 待验证假设 / 待定决策

> **本节状态：OPEN-ASSUMPTIONS / EXTERNAL-NOT-MEASURED；这些项目是生产资格前置验证，不是当前完成声明。**


1. **实测吞吐**:拿 1 小时真实视频在 7B + vLLM 上测 tokens/秒 与 GPU-小时,校准 4.3 的估算再签 SLA。
2. **7B 是否够用**(QA 兜底判断 + 预标注草稿),鱼眼画面上验证。
3. **Stage 1 自信清除率**(QA 能放行多少不进 VLM),决定 QA token 降幅。
4. **动词聚类方法**:人工映射表 / LLM 辅助 / embedding 聚类。
5. **搜索 embedding**:文本优先 vs 视觉(倾向文本先起步、视觉后加)。
6. **pod 网络带宽**:查 RunPod 规格并实测 R2→pod 速度。

## 6. 分期建议

> **本节状态：PROPOSED-ROADMAP / 非承诺；仅供当前 Blueprint 和实施窗口参考，不是审批流程或交付承诺。**


按依赖顺序排列(标出依赖关系):

1. **QA 逐 clip 打标 + pass/warning/fail**(数据模型 + 审片 UI 起止点)—— QA 工作线,产出**问题段**。
   注意:这是 QA,**不是动作分割**。
2. **动作分割产出**(Assignment 1 标注 / AI 预标注)—— 按 annotation principal 产出 `structured_labels`
   (动作-物体分段),**是搜索的输入**。
3. **搜索 MVP**(动词聚类 + clip 索引 + faceted,零 GPU)—— **依赖第 2 步的动作分割输出**。
   可先在现有已标注样本上跑通"clip 索引 + 跳播"链路,不必等生产管线。
4. **生产管线上云**(R2 + Supabase + vLLM 化的 GPU 任务,抽帧共享)—— 让 QA + 预标注在 500h/天达标。
5. **搜索 embedding 阶段**(文本向量 → 视觉向量)。

---

## 附录 A —— 21 类质检问题清单

> **本节状态：REFERENCE-VOCABULARY / 非权威副本；执行词汇以源码、发布 schema、测试和 conformance fixture 为准。**


平台质检下拉的全部问题(权威源为 `app.js` 的 `QA_ISSUE_GROUPS`,由 `ai_qa/check_vocabulary.py` 校验一致)。

**Device Issues / 设备问题**
1. Black Screen — 黑屏
2. Glitched Screen — 画面损坏
3. Blurry Lens — 镜头模糊

**Collector Operation Issues / 采集员操作问题**
4. Excessive Speed — 移动过快
5. Ego - Device worn backwards — 设备戴反
6. Ego - Hand not centered in frame — 手不在画面中心
7. Camera stationary for more than 5s — 相机静止超过 5 秒
8. Hair blocking view — 头发遮挡视线
9. Irrelevant actions in partial segments — 局部片段含无关动作
10. Task irrelevant actions — 任务无关动作
11. Arm/Hand obstructed — 手臂/手被遮挡
12. Hand overlap / contact / crossing — 双手重叠/接触/交叉
13. Incomplete task — 任务未完成
14. Lack of diversity — 缺乏多样性
15. Lack of authenticity — 真实性欠缺
16. Video Abnormally Ending — 视频异常结束

**Environmental Issues / 环境问题**
17. Too Dark / Overexposed — 过暗/过曝
18. Unauthorized Person/Animal Entering Frame — 无关人员/动物入镜
19. Revealing outfit — 着装暴露

**Task Set Issues / 任务集问题**
20. Performed other existing Tasks — 执行了其他已有任务

**Others / 其他**
21. Other (please specify) — 其他(请注明)

---

## 附录 B —— 片段处理与后处理设计

> **本节状态：REFERENCE-DESIGN / 非权威；设计背景可复用，但不得覆盖当前 Blueprint 或 executable contract。**


### B.1 原则

> **本节状态：REFERENCE-DESIGN / 非权威；仅记录设计意图。**


1. **不物理剪源。** 原始 `.mcap` 永远不动 —— 留 provenance、可重新质检、可撤销。有效性以**时间段标记**表达,不改源。
2. **有效性是"按时间段 + 按用途"的**,不是一条片子一个全局 pass/fail。同一段可能对手部关键点无效、对整体操作序列有效。
3. **QA 在标注之前。** 剪/标发生在 QA 阶段,标注是对已处理 footage 做的,所以不存在标注脱钩。
4. **新增派生量:有效时间。** `effective_duration = raw_duration − Σ(无效段时长)`。不改 manifest 的 `duration`(设备原始记录,source of truth)。raw 和 effective **都显示** —— raw 是采集员工作量,effective 是数据产出。业界对应:Unidata 用"每小时可用分钟数"做主指标,预期 30–40% 素材通不过质检。

### B.2 21 个问题的分类(完整映射)

> **本节状态：REFERENCE-DESIGN / 非权威映射；运行时行为必须由代码与测试证明。**


判据一根轴:**缺陷是"卡在某段时间、周围可用"→ 打标(warning);是"整条录制的属性 / 缺了某样东西 / 合法性合规问题"→ 整片拒(fail)。**

| 处理 | 问题 | 原因 |
|---|---|---|
| **打标 warning(时间局部)** | Black Screen 黑屏 * | 通常只黑一段 |
| | Glitched Screen 画面损坏 | 解码损坏多为一段 |
| | Excessive Speed 移动过快 | 快是某一段 |
| | Ego - Hand not centered 手不在中心 | 手飘出去一阵 |
| | Camera stationary >5s 静止超 5 秒 | 本就是一个时间窗 |
| | Hair blocking view 头发遮挡 | 头发扫过一阵 |
| | Irrelevant actions in partial segments | "partial" 本身即时间局部 |
| | Task irrelevant actions 任务无关动作 | 跑题一段,其余在任务上 |
| | Arm/Hand obstructed 手臂/手被遮 | 遮一段(手部 QA 主场) |
| | Hand overlap / contact / crossing | 一个瞬间/一段 |
| | Too Dark / Overexposed 过暗过曝 * | 光照变化多为一段 |
| | Unauthorized Person/Animal 无关人/动物入镜 | 走过一阵,标那个窗 |
| **整片拒 fail(整条属性)** | Ego - Device worn backwards 戴反 | 全程视角错 |
| | Blurry Lens 镜头模糊 | 镜头脏/虚是全程 |
| | Incomplete task 任务未完成 | 缺的是结尾,没有段可标 |
| | Lack of authenticity 真实性欠缺 | 整段表演,合法性问题 |
| | Revealing outfit 着装暴露 | 全程 + 合规,剪几秒没用 |
| **修尾(留前半)** | Video Abnormally Ending 异常结束 | 截断点前可用,标/剪坏尾巴 |
| **不可判(不是按片打标)** | Lack of diversity 缺乏多样性 | 要跨多条片子看 |
| | Performed other existing Tasks 做成了别的任务 | 要任务目录;整片标错→重指派 |
| **看情况** | Other 其他 | 视具体而定 |

\* 带星号:若**整条**都黑/都过曝(非一段),归 fail。判据始终是"时间局部 + 周围可用"。

规律:**"采集员操作"类多是时间局部 → 偏 warning;"设备/合规/完整性"类多是整条属性 → 偏 fail。**

### B.3 QA 阶段怎么做

> **本节状态：REFERENCE-DESIGN / 非权威；不能替代本地 conformance 或生产资格报告。**


- **打标类(warning)**:质检员记录 `{start_time, end_time, issue_type}` —— 一个**时间段**,不只是片级判定。
- **整片拒类(fail)**:整条 QA fail,不进标注,删除。
- **修尾类**:把尾部标成无效段(位于结尾的 range)。
- **不可判类**:在别的层处理(跨片 / 任务目录),不是逐片 QA。
- 源不动,标记作为元数据挂在旁边;算出 `effective_duration`。

### B.4 后处理怎么做(SLAM + 骨骼,短 vs 长)

> **本节状态：PROPOSED / 外部依赖；需要传感器、模型和代表性数据验证。**


QA 的 `{时间段, 问题类型}` 标记是后处理的**输入**。后处理**按每个坏段、分信号**决策,由**坏段时长**驱动。
关键:**同一个坏段,SLAM 和骨骼可以做不同选择,因为 IMU 只救得了 SLAM。**

- **IMU 是独立传感器**(物理测运动,不看图像)—— RGB 坏的那段它还是真数据,能给 SLAM 当桥。
- **骨骼派生自 RGB**(HaMeR 读画面出关节)—— RGB 坏,骨骼跟着坏,**没有独立的桥**,只能补全(猜)。

| 坏段 | SLAM(有 IMU 桥) | 骨骼(无独立桥) |
|---|---|---|
| **短(约 1s 内)、视觉原因、IMU 好** | IMU 递推桥接,**保持连续**(短时漂移厘米级) | 前后插值补全 + **标低置信**,保持连续 |
| **长** | 设硬段界**拆段**;能重定位就重拼,否则当两段 | 留**空洞** / 硬段界,**不补**(补出来是编的) |
| **黑屏 / 全丢** | 硬拆 | 硬段界 |
| **坏段在开头/结尾** | 直接修掉那头,不用桥 | 同上 |

**补充判断:**
- **"RGB 差"分原因** —— 遮挡(补全);太暗/模糊(可去模糊/提亮重抽出**真**姿态,未必补);黑屏(视觉全丢)。
- **长坏段绝不把补全当真数据发出**;短遮挡补得住,长遮挡这段手可能干了任何事。
- **补全的置信度**属于"补出来的姿态值",不是"QA 判 fail"。QA 只打标,不补;补是下游想要连续骨骼轨迹时才做。

### B.5 对机器人学习的影响

> **本节状态：BACKGROUND / 非权威；仅作数据治理和后处理背景。**


- **模仿学习(这批数据最可能的用途)对拆段基本无害** —— 分段独立训练本是常态,只要**不跨段界训练**。
- **遮挡未必是缺陷** —— 真实操作本就有遮挡,策略常对它鲁棒;带标记留着有时比剪掉更好。逐帧深度不受剪口影响。

### B.6 业界对照(印证)

> **本节状态：BACKGROUND / 非证据；业界材料不能替代仓库测试或生产测量。**


- **Unidata(生产实践)**:遮挡"打标 + 置信度",优先于删除;"每小时可用分钟数"做主指标;当场离场前立即质检。
- **Ego4D / Ego-Exo4D**:拒绝标签 + 阈值过滤整片。
- **UMI**:手持相机跑视觉 SLAM,对遮挡/丢跟踪脆弱,拿不到位姿就丢数据;UMI-3D 加 LiDAR、EgoDex(Vision Pro)设备端实时出位姿 —— 都是"加/留一路非视觉信号桥接",印证"别把 IMU 一起剪掉"。
