# Phase 1 测试历程与问题闭环

日期：2026-09-28  
项目：CachePilot  
范围：渐进压测、策略对照、故障注入、Phase 1 出口、并发失败定位

本文记录截至 2026-09-28 已执行的全部相关测试、原始结果、问题判断、修复动作和
下一次重跑方式。本文区分“测试通过”和“具备 Phase 1 验收证据”：前者只说明某个
路径工作，后者还要求真实执行器、固定硬件/模型、完整 artifacts 和重复协议。

更新说明：第 1–12 节保留早期 RTX 4080 SUPER 调试历程，其中“当前”“待重跑”仅指
当时状态。最新 Colab T4 结果、分析更正和下一轮闭环见第 13 节及后续章节。

## 1. 验收目标

Phase 1 需要同时证明：

1. 单 GPU 真实执行器可以完成 API/SSE/取消/超时路径。
2. Strict/Adaptive、FCFS/WFQ、prefix-blind/prefix-aware 在相同执行器、模型、硬件和
   trace 内可重复比较。
3. 渐进压测能区分成功、过载保护、超时、执行失败和 OOM，而不是把鉴权失败当成容量边界。
4. 取消、断连、执行超时、执行器异常和 OOM 都只产生一个终态，并释放 reservation。
5. Phase 1 报告包含 TTFT、TPOT、P99、吞吐、公平性、拒绝率、KV 峰值、GPU 秒和估算成本。

## 2. 测试前置条件

GPU 服务器使用以下环境出口：

```bash
python --version
nvidia-smi
python main.py --check
```

已拉回的真实 GPU progressive report 记录了：

- Python `3.13.15`
- Linux x86_64
- 单 GPU：`NVIDIA GeForce RTX 4080 SUPER`
- 显存约 `16 GiB`
- CUDA `13.0`
- PyTorch `2.11.0+cu130`
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- model/tokenizer revision：`7ae557604adf67be50417f59c2c2f167def9a775`
- context limit：`8192`

这部分环境条件满足单卡实测的基本门禁。

## 3. CPU/SimExecutor 基线测试

### 3.1 已有命令和目的

仓库已有 CPU 单元/契约测试，覆盖状态机、Registry、资源租约、Strict/Adaptive admission、
FCFS/WFQ、SimExecutor、RuntimeLoop、Gateway、账本和协议分析器。Phase 0 的统一入口是：

```bash
python benchmarks/phase0_exit.py
```

相关专项测试包括：

```bash
python -m unittest \
  tests.unit.test_admission \
  tests.unit.test_adaptive_admission \
  tests.unit.test_scheduler \
  tests.unit.test_sim_executor \
  tests.unit.test_runtime_loop \
  tests.integration.test_gateway_api \
  tests.integration.test_vllm_gateway
```

本地开发机没有安装 `pytest`/FastAPI，因此本轮没有在本地重新执行完整测试套件；代码修改后
执行了 Python 编译检查和 `git diff --check`。GPU 服务器应在虚拟环境中重新运行上述命令。

### 3.2 已有 Sim 结果

以下目录已存在完整的 Sim artifacts：

```text
runs/demo-sim-mixed-001/
runs/strict-fcfs/
runs/adaptive-fcfs/
runs/strict-wfq/
runs/adaptive-wfq/
```

这些 run 都是：

- `executor=SimExecutor`
- `gpu_count=0`
- `warmup=false`
- `repetition_index=0`
- `trace_id=mixed-length-seed-7`
- `seed=7`

主要 summary 数值一致：

| 指标 | 值 |
|---|---:|
| queue P99 | 15 ms |
| TTFT P99 | 21 ms |
| TPOT P99 | 0.2258 ms |
| total P99 | 38 ms |
| completion throughput | 1578.95 token/s |
| Jain fairness | 0.8427 |
| rejection rate | 0 |
| cancellation rate | 0 |
| reserved KV peak | 16 blocks |

### 3.3 结果解释

这些结果证明 SimExecutor 和分析器可以生成可重放的模拟摘要，但不能作为 GPU 性能结论。
四个策略在这份小型 mixed-length trace 上数值完全相同，可能原因是：

- trace 太短，未形成 FCFS/WFQ 的可区分队列竞争；
- 没有 prefix-aware 运行；
- 没有 warm-up 和三次测量；
- Sim 结果反映逻辑时钟，不等同于真实 GPU wall-clock。

因此不能得出“Strict=Adaptive”或“FCFS=WFQ”的性能结论。

## 4. 第一次渐进压测：全部 403

### 4.1 使用的命令

第一次使用了类似命令：

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --output-dir runs/progressive-strict-fcfs-blind \
  --contexts 32,128,512,2048 \
  --concurrencies 1,2,4 \
  --max-tokens 16 \
  --repetitions 3 \
  --warmup
```

脚本原先固定发送：

```text
X-Tenant-ID: progressive-load
```

### 4.2 结果

以下目录的每个点全部是 HTTP 403：

```text
runs/progressive-20260928-222406/
runs/progressive-adaptive-fcfs-blind/
runs/progressive-full-20260928-222629/
runs/progressive-full/
runs/progressive-strict-fcfs-blind/
runs/progressive-strict-wfq-blind/
runs/progressive-strict-wfq-aware/
```

表现为：

- `safe_context_tokens_observed=null`
- `first_overload_protection_point=null`
- `first_oom_point=null`
- 所有点 `counts={"rejected": ...}`

### 4.3 问题和判断

这是 Gateway tenant 鉴权失败，不是模型拒绝、过载保护或 OOM。服务默认只允许配置过的
tenant（例如 `team-a`、`team-b`），`progressive-load` 不在授权列表中。

### 4.4 修复

`benchmarks/progressive_load.py` 已增加参数：

```bash
--tenant-id team-a
```

该参数同时用于提交请求和查询 trace，默认值为 `team-a`。这样不会再因为脚本内部硬编码
未授权 tenant 而产生无效容量报告。

## 5. 第二次渐进压测：单并发成功，并发 2/4 失败

### 5.1 有效重跑命令

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --tenant-id team-a \
  --output-dir runs/progressive-strict-fcfs-blind-v2 \
  --contexts 32,128,512,2048 \
  --concurrencies 1,2,4 \
  --max-tokens 16 \
  --repetitions 3 \
  --warmup
```

### 5.2 环境结果

该报告已经是真实 GPU 测试，不再是 403：

- GPU：RTX 4080 SUPER
- GPU count：1
- model/tokenizer revision 固定
- readyz：HTTP 200

### 5.3 结果

并发 1 的四个 context 点全部三次成功：

| context target | 实际 prompt tokens | concurrency | 结果 |
|---:|---:|---:|---|
| 32 | 62 | 1 | 3/3 success |
| 128 | 158 | 1 | 3/3 success |
| 512 | 542 | 1 | 3/3 success |
| 2048 | 2078 | 1 | 3/3 success |

报告给出的安全点是：

```text
safe_context_tokens_observed=2078
safe_concurrency_at_safe_context=1
```

并发 2 和 4 出现 HTTP 500；整体统计为 47 success、37 failed。失败不是 429、timeout 或
OOM，而是 `internal_error` 和 `executor_failed`。

### 5.4 第一次猜想

初步怀疑是：

- TorchExecutor 不能安全并发调用模型；
- Gateway 新接入 Scheduler 后的状态/队列竞争；
- tokenizer 与模型调用存在共享状态竞争。

## 6. 并发 2 最小复现

### 6.1 命令

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --tenant-id team-a \
  --output-dir runs/progressive-debug-c2 \
  --contexts 32 \
  --concurrencies 2 \
  --max-tokens 16 \
  --repetitions 3 \
  --warmup
```

### 6.2 结果

三次重复均为一成功一失败。成功请求大约为：

- TTFT：约 265–276 ms
- total：约 267–278 ms
- completion：16 tokens

失败请求为 HTTP 500，并出现：

```text
error_code=executor_failed
error_code=internal_error
```

### 6.3 根因定位

服务器 traceback 显示：

```text
File cachepilot/gateway/api.py, line 604, in query
  prompt_tokens = self.token_counter.count_prompt_tokens(snapshot.request)
File cachepilot/executors/torch_executor.py, line 132, in count_prompt_tokens
  token_ids = apply_template(...)
RuntimeError: Already borrowed
```

根因是 Transformers tokenizer 的内部对象被两个并发路径同时使用：一个请求在生成，另一个
请求查询 `/v1/requests/{id}` 并重复调用 tokenizer。此错误发生在 tokenizer 层，不是 GPU
显存不足。

### 6.4 修复

`cachepilot/executors/torch_executor.py` 已增加 `_tokenizer_lock`，并覆盖：

- `count_prompt_tokens()` 的 chat template/tokenize；
- `_encode()` 的 chat template 和输入编码。

`_generation_lock` 仍然保护模型 `generate()`，新增 tokenizer 锁解决 tokenizer 的独立
并发问题。

### 6.5 修复后的重跑

GPU 服务器必须同步最新代码并重启服务：

```bash
pkill -f 'python main.py' || true

python main.py \
  --admission strict \
  --scheduler fcfs \
  --prefix-mode blind \
  2>&1 | tee runs/server-strict-fcfs-blind-v3.log
```

然后重跑：

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --tenant-id team-a \
  --output-dir runs/progressive-debug-c2-v3 \
  --contexts 32 \
  --concurrencies 2 \
  --max-tokens 16 \
  --repetitions 3 \
  --warmup
```

验收条件：

```text
counts={"success":2}
safe=true
completed_repetitions=3
```

## 7. 故障注入测试

### 7.1 命令

```bash
python -m benchmarks.chaos_validation \
  --output runs/phase1-chaos/chaos_report.json
```

### 7.2 结果：软件注入 PASS

`runs/phase1-chaos/chaos_report.json` 的 `status` 为 `PASS`，五个 case 均只有一次终态转换，
且 `reserved_blocks=0`、`active_sequences=0`：

| case | terminal state | HTTP | error code | terminal transitions |
|---|---|---:|---|---:|
| cancel | `CANCELLED` | — | — | 1 |
| disconnect | `CANCELLED` | — | — | 1 |
| timeout | `TIMED_OUT` | 504 | `deadline_exceeded` | 1 |
| exception | `FAILED` | 500 | `executor_failed` | 1 |
| oom | `FAILED` | 500 | `executor_oom` | 1 |

### 7.3 限制

这份报告证明的是进程内故障注入和资源回收，不是物理 GPU OOM，也不是客户端真实 TCP/HTTP
断连。真实 GPU OOM 和真实 SSE 客户端中途断开仍需在服务器上单独保存日志和请求记录。

## 8. 策略对照准备和当前缺口

### 8.1 策略启动参数

当前服务支持：

```bash
python main.py --admission strict --scheduler fcfs --prefix-mode blind
python main.py --admission adaptive --scheduler fcfs --prefix-mode blind
python main.py --admission strict --scheduler wfq --prefix-mode blind
python main.py --admission strict --scheduler wfq --prefix-mode aware
```

这些参数已接入 Gateway 的 admission/scheduler 路径。`prefix-aware` 需要真实逻辑 prefix
命中 token；当前请求契约尚未携带 tokenized prefix，因此没有命中时会退化为 WFQ 基线。

### 8.2 现有对照 artifacts 的问题

`runs/strict-fcfs`、`runs/adaptive-fcfs`、`runs/strict-wfq`、`runs/adaptive-wfq` 都是
SimExecutor 单次 run，且都为 prefix-blind。缺少：

- prefix-aware run；
- 每策略 warm-up；
- 每策略至少三次 measured run；
- 真实 TorchExecutor/vLLM run；
- 相同 GPU 上的完整策略矩阵。

运行：

```bash
python -m benchmarks.strategy_matrix \
  --metric ttft_ms \
  --output runs/required-matrix/matrix.json \
  runs/<all-strategy-run-directories>
```

当前对已有四个 Sim run 执行时得到：

```text
MATRIX_INVALID: missing required prefix_mode pair: blind vs aware
```

这是正确拒绝，不应绕过。

## 9. Phase 1 出口检查

### 9.1 命令

```bash
python -m benchmarks.phase1_exit \
  --chaos-report runs/phase1-chaos/chaos_report.json \
  --output runs/phase1-exit/report.json \
  runs/<sim-run> \
  runs/<torch-or-vllm-run>
```

### 9.2 当前失败原因

用现有旧 Sim run 尝试时得到：

```text
PHASE1_EXIT=FAIL:
runs/demo-sim-mixed-001: estimated GPU seconds and cost are required
```

这说明旧 `requests.jsonl` 没有完整保存：

```text
estimated_gpu_seconds
estimated_cost
```

此外当前还缺少真实 measured executor run，因此 Phase 1 出口不能通过。

## 10. 当前状态总表

| 项目 | 当前状态 | 证据 |
|---|---|---|
| GPU 环境门禁 | PASS | progressive report environment |
| Sim/CPU 基线 | 已有，但仅模拟 | `runs/*summary.json` |
| 第一次 progressive load | INVALID | 全部 403 tenant rejection |
| 修复 tenant 后 progressive | 部分通过 | concurrency 1 成功；2/4 HTTP 500 |
| 并发根因 | 已定位 | tokenizer `Already borrowed` traceback |
| tokenizer 修复 | 已提交，待服务器重跑 | `_tokenizer_lock` |
| 故障注入 | PASS（软件注入） | `runs/phase1-chaos/chaos_report.json` |
| 真实 GPU OOM | 未完成 | 需要物理 GPU 证据 |
| Strict/Adaptive GPU 对照 | 未完成 | 缺完整重复矩阵 |
| FCFS/WFQ GPU 对照 | 未完成 | 缺完整重复矩阵 |
| prefix-blind/aware | 未完成 | 缺真实 prefix-aware 命中证据 |
| Phase 1 出口 | FAIL | 缺完整 ledger 和真实 measured run |

## 11. 推荐的下一轮顺序

### 11.1 先验证 tokenizer 修复

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --tenant-id team-a \
  --output-dir runs/progressive-debug-c2-v3 \
  --contexts 32 \
  --concurrencies 2 \
  --max-tokens 16 \
  --repetitions 3 \
  --warmup
```

### 11.2 再跑完整渐进矩阵

只有最小并发复现 `safe=true` 后才运行：

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --tenant-id team-a \
  --output-dir runs/progressive-strict-fcfs-blind-v3 \
  --contexts 32,128,512,2048,4096,8192 \
  --concurrencies 1,2,4,8 \
  --max-tokens 16 \
  --repetitions 3 \
  --warmup
```

### 11.3 每个策略重复

每个策略都要保存至少一次 warm-up 和三次 measured run，然后对每个 run 执行：

```bash
python benchmarks/analyze.py \
  --manifest runs/<run>/manifest.json \
  --trace runs/<run>/trace.jsonl \
  --requests runs/<run>/requests.jsonl \
  --output runs/<run>/summary.json
```

### 11.4 最后执行门禁

```bash
python -m benchmarks.strategy_matrix \
  --metric ttft_ms \
  --output runs/required-matrix/matrix.json \
  runs/<all-required-runs>

python -m benchmarks.phase1_exit \
  --chaos-report runs/phase1-chaos/chaos_report.json \
  --output runs/phase1-exit/report.json \
  runs/<sim-run> runs/<real-executor-run>
```

## 12. 结论

目前最重要的技术问题已经定位到 tokenizer 并发访问，并已用 `_tokenizer_lock` 修复。之前
的 500 不能作为模型容量边界。只有修复后的 `progressive-debug-c2-v3` 在三次重复中全部
成功，才可以继续把 concurrency 2/4/8 纳入安全矩阵。

截至本文记录时，Phase 1 仍不能标记完成：故障注入软件路径已通过，但真实 GPU OOM、真实
断连、完整四策略重复对照、真实执行器 summary 和 Phase 1 exit report 仍需补齐。

## 13. Colab T4 完整验证：20260928T193447-d1eadc

### 13.1 证据与环境

原始包为 `20260928T193447-d1eadc-evidence.zip`，本机下载位置为
`/Users/ys/Downloads/20260928T193447-d1eadc-evidence.zip`。分析时直接读取 ZIP，未修改
原始实验结果。398 个文件的 checksums 校验一致；12 个 warm-up、12 个 measured 和
1 个 Sim 的 25 份 summary 重新计算全部一致，代码快照与 source.json 的哈希一致。

ZIP SHA-256：`ac0808f73828e1c98b8072aa3189580c93818abb81da96f12817f2747ea2c7f6`。

- Python 3.13.15，PyTorch 2.11.0+cu130，CUDA 13.0，驱动 580.82.07。
- 单张 Tesla T4，报告显存 15 GiB；Qwen/Qwen2.5-0.5B-Instruct，固定 revision
  `7ae557604adf67be50417f59c2c2f167def9a775`。
- **实际 dtype=bfloat16**，不能把本轮写成 T4/float16 结果。
- 代码 commit `f44c35d3c17784fba272b0003925bb1e14c7464d`，工作树 dirty；复现以包内
  `source-snapshot/` 和文件哈希为准，不能只依赖 commit。
- Phase 0 在 Colab 实际执行 185 项测试全部通过；所有步骤退出码为 0。

### 13.2 四项验收结论

| 项目 | 本轮结论 | 解释 |
|---|---|---|
| 渐进压测 | PASS（限已测配置） | 四策略完成扫描，保护边界可重现；未测出物理显存极限 |
| 必要对照 | INCONCLUSIVE | 原始矩阵齐全，但策略触发及 workload 区分度不足 |
| 故障验证 | PASS（受控注入） | 五项唯一终态、资源回收及正常请求恢复通过 |
| Phase 1 出口 | NOT_PASSED | 必要对照尚未闭环 |

`phase1-gate.log` 的 `PHASE1_EXIT=PASS` 是旧脚本的局部字段/故障门禁；它没有检查
策略是否真正触发。最终应以 `conclusions.json` 的 `NOT_PASSED` 为准。

### 13.3 渐进压测与容量口径

每策略 6 个 context × 5 个并发 × 3 次重复，共 90 个 step、558 条非 warm-up 请求。
四策略完全相同：258 成功、300 个 HTTP 429；没有自然 OOM、超时或执行失败。

| prompt 目标长度 | 实际 prompt tokens | 全量重复成功的最高受测并发 |
|---:|---:|---:|
| 32 | 62 | 8 |
| 128 | 158 | 8 |
| 512 | 542 | 8 |
| 2048 | 2078 | 2 |
| 4096 | 4126 | 1 |
| 8192 | 8222 | 无，超过 context limit |

每策略的 429 原因为：`max_active_sequences` 72 次、`kv_capacity` 135 次、
`context_limit_exceeded` 93 次。短请求并发 16 触发服务配置的 active=8；长请求先
触发 512−32=480 个可用逻辑 block 的限制；8192 目标长度加模板后超过 8192。
这些是控制层配置边界，不是 T4 的物理 OOM 边界。表中并发是 HTTP 请求并发，
不是 Torch continuous batching；不能将不同点的最大 context 和并发拼成安全配置。

### 13.4 正式对照结果与解释

每策略 3 次正式测量，每次 32 请求，均为 8 FINISHED + 24 REJECTED；拒绝全部源于
`max_active_sequences`。12 次正式 run 合计 96 完成、288 拒绝，拒绝率 75%。

| 策略 | TTFT P95 中位数（秒） | 窗口吞吐中位数（token/s） | 拒绝率 |
|---|---:|---:|---:|
| strict/fcfs/blind | 9.276 | 27.51 | 75% |
| adaptive/fcfs/blind | 9.308 | 27.39 | 75% |
| strict/wfq/blind | 9.368 | 27.21 | 75% |
| strict/wfq/aware | 8.662 | 29.33 | 75% |

1. **Adaptive 原因更正**：首次人工分析声称 Gateway 没有反馈输出历史，该判断有误。
   完整链路是 `complete/stream → _complete_admission → admission.complete`，已经存在。
   本轮 warm-up 全部实际生成 32 tokens，等于 max_tokens；因此
   `min(max_tokens, P95 + safety_margin)` 仍为 32，无法减少预留。active 硬限制又优先
   阻断请求。不能从指标近似推出 Adaptive 算法无效，需记录 fallback/估计值/历史量。
2. **Prefix 接线缺失**：账本全部 `logical_hit_source=not_configured`、logical_hit=false。
   Gateway 没有查询 PrefixIndex，也未将命中 token 传给 SchedulingRequest，aware
   实际退化为无命中 WFQ。表面吞吐改善约 7.8% 不能归因于 prefix 优化；部分 run
   实际被接纳的 prompt 组成也有差异。
3. **调度竞争不足**：同权 tenant、固定 service_cost=32，75% 请求在调度前被拒绝；
   调度器未以 Torch 执行槽限制 dispatch，真实等待主要落在模型锁。接近 1 的 Jain
   指数不能证明 WFQ 有效；需要可见的调度队列等待、选择顺序和不同成本负载。
4. **时间/成本限制**：Torch 整段 generate 后才交付 token，约 0.2 ms 的 TPOT 是交付
   指标，不是 GPU decode 速度。成本全部 null，代表未配置价格，不代表免费。

### 13.5 真实故障证据

| 故障 | 终态 | 唯一终态次数 | reservation/active | 恢复 HTTP |
|---|---|---:|---|---:|
| 显式及重复取消 | CANCELLED | 1 | 0 / 0 | 200 |
| SSE 断连 | CANCELLED | 1 | 0 / 0 | 200 |
| 执行超时 | TIMED_OUT | 1 | 0 / 0 | 200 |
| model.forward 异常 | FAILED | 1 | 0 / 0 | 200 |
| CUDA allocator OOM | FAILED | 1 | 0 / 0 | 200 |

五项 allocated bytes 都回到模型基线 1,005,890,560，取消原因分别为
explicit/disconnect/timeout。OOM 有 forward_entered 和 cuda_oom_observed 两项
原始证据，Gateway 映射为 executor_oom。OOM 是 model.forward 内主动超容量分配，
不是自然负载 OOM，也没有覆盖进程被 kill 或长期重复故障。

### 13.6 本轮流程问题闭环

- Colab 排除 runs 后，诊断测试读取历史样例失败：固定输入迁到
  `tests/fixtures/diagnostics/`；新增文件必须与测试代码一起同步。
- 重复执行 notebook 丢失旧服务句柄、8000 被占用：初始化/服务管理单元先清理
  已记录的服务；未知占用者不能直接 kill。
- T4 误选 bfloat16：默认 `is_bf16_supported()` 可包含模拟支持，下一轮要求原生
  BF16 支持，否则固定 float16。新旧 dtype 数据不得混合组成对照。

## 14. 下一轮闭环计划

| 工作 | 验收要求 | GPU 状态 |
|---|---|---|
| Adaptive 可观测性与回归 | 短输出历史降低预留，stream/non-stream 均学习，增长不破硬上限 | 待新一轮 |
| PrefixIndex → Scheduler → Ledger | 同 tenant/version 真实 token 命中、隔离成立、物理命中仍 null | 待新一轮 |
| Torch 调度执行槽 | 等待在 Scheduler 可见，取消/超时不泄漏队列或槽位 | 待新一轮 |
| 有区分度的对照 workload | 避免固定 75% 拒绝，输出短于预算，混合成本并保存策略触发证据 | 待新一轮 |
| T4 dtype | 原生支持检测，T4 float16，环境/manifest 一致 | 待新一轮 |

本地修复及回归不代替 Colab GPU 验收；保留本轮证据，下一轮使用新 session 完整重跑。

### 14.1 已实现的代码闭环

- 保留已有 `_complete_admission` 学习链路，新增端到端测试证明非流式和 SSE 完成
  都更新历史；`GET /v1/requests/{id}` 的 `policy` 返回初始输出估计和 fallback，
  `adaptive` 返回历史桶量及估计误差。实际成本依然遵守估算口径。
- Adaptive 长尾增长同时更新 Controller、Registry 和 telemetry 的 reservation，
  硬容量不足时进入 FAILED 并释放资源，不遗留 EXECUTING 或失真的 KV 峰值。
- Torch 通过同一 tokenizer 锁取得真实 token IDs，生成带 tenant/model/tokenizer/
  quantization 作用域的 PrefixScopeKey。Gateway 按最近完成顺序保留最多 256 个唯一
  逻辑 key，下一次提交查询命中；blind/aware 都观测命中，只有 aware
  将命中传给调度器。账本来源为 `tenant_prefix_index`，physical_hit 仍为 null。
- Torch 保留 8 个逻辑 active reservation，但只发放 1 个执行槽。未获槽位的
  已准入请求停留在 Scheduler；按 prompt+max_tokens 作为工作量，记录选择顺序、
  `scheduler_wait_ms` 和 `cache_boosted`。完成、异常、等待中超时/取消都会清理槽位。
  admission 容量不足仍按既有 Gateway 行为拒绝，未在本次改造为完整准入等待队列。
- 服务策略版本升级为 `gateway-policy-v2`；正式 manifest 从实际 ledger 读取版本。
- server 与 notebook 均使用 `is_bf16_supported(including_emulation=False)`，使
  T4 选择 float16；不接受模拟 BF16 支持冒充原生支持。

### 14.2 第二轮 Colab 协议

继续使用 `notebooks/colab_phase1_matrix.ipynb`，新建 session，同步全部代码和新增
测试文件。环境/API/渐进扫描/故障/Sim/归档的主流程不变，正式 workload 更新为：

```text
profile=mixed-policy-v2
contexts=[1024,1024,512,512] × 16 = 64 requests/run
tenants=team-a,team-b
max_tokens=1024 / 128
instructions=只回复 OK / 列出 1..12，不以生成达到上限为目标
concurrency=8, wave_size=8, execution_slots=1
warmup=每轮重启后串行同一 workload
repetitions=3/strategy
```

每波结束才发下一波，避免一个拒绝快速触发后续全部请求；保留每个真实输入、实际
发送/完成时间、策略诊断。trace_id 哈希覆盖输入文本、token trace 和 profile，
不只比较 trace 名称。闭环分波负载的实际到达时间随完成时间变化，不能写成严格的
开放环固定时间回放；吞吐用实际观测窗口计算。真实模型未遵循短输出要求时仍须
判定对照不足，不能修改输出 token 或把拒绝排除出统计。

新增 `policy-coverage.json`，每个正式 run 至少 24 完成、拒绝率不超过 50%、
Scheduler 等待 P95 至少 1 ms；每个 Adaptive run 需非 fallback 且有预留缩减；
每个 prefix-aware run 需有逻辑命中和实际 dispatch boost。任一条件缺失，必要
对照为 INCONCLUSIVE，最终出口不通过。以上是实验是否触发策略的覆盖门槛，
不是性能 SLO 或“必须有收益”的要求。

### 14.3 本地验证与尚未完成项

本地 Python 3.12.13 CPU 回归：`python -m pytest -q`，204 passed，1 skipped
（沙箱不允许绑定本地端口）。新增测试覆盖 Adaptive 学习/增长/硬限额释放、
tokenized prefix 的 tenant/version 隔离、实际调度 boost、FCFS/WFQ 顺序差异、
单执行槽及排队超时/取消、分波提交不提前启动下一波、原生 BF16 检测。
改动涉及的 Python 文件通过 Ruff，`git diff --check` 通过；notebook 代码单元
语法检查及“不足证据不能通过出口”的回归也通过。

这些是代码和实验流程的本地验证，不是锁定 Python 3.13.15 的 GPU 复验。
第二轮 Colab 尚未执行，TODO 的必要对照与 Phase 1 出口继续保持未完成。
