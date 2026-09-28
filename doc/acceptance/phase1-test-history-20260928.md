# Phase 1 测试历程与问题闭环

日期：2026-09-28  
项目：CachePilot  
范围：渐进压测、策略对照、故障注入、Phase 1 出口、并发失败定位

本文记录截至 2026-09-28 已执行的全部相关测试、原始结果、问题判断、修复动作和
下一次重跑方式。本文区分“测试通过”和“具备 Phase 1 验收证据”：前者只说明某个
路径工作，后者还要求真实执行器、固定硬件/模型、完整 artifacts 和重复协议。

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
