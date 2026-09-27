# CachePilot 单机运行手册

本手册适用于一个 Gateway 进程绑定一个执行器的单机部署。它不假设多副本自动切换，
所有阈值必须以同模型、同执行器、同硬件的已验收基线和业务 SLO 为准，不能把本文示例
当作通用容量结论。

## 1. 先保全证据

故障处置前先停止继续加压，并导出当前 `/metrics` 与对应实验的原始产物。导出器只使用
Python 标准库，不需要本机部署 Prometheus：

```bash
make ops-export \
  BASE_URL=http://127.0.0.1:8000 \
  RUN_DIR=runs/<run_id> \
  OUTPUT_DIR=artifacts/<run_id>-incident
```

`RUN_DIR` 必须已有协议 v1 的 `manifest.json`、`trace.jsonl` 和 `requests.jsonl`。命令会先
校验三份原始产物，再生成：

| 文件 | 用途 |
|---|---|
| `metrics.prom` | 未加工的 Prometheus 文本快照 |
| `snapshot.json` | 抓取时间、来源 URL、run ID 和快照 SHA-256 |
| `summary.json` | 机器可读实验汇总 |
| `report.md` | 单机在线状态与实验 P50/P95/P99 报告 |

导出失败不得用旧快照冒充现场结果。先检查 `curl -fsS http://127.0.0.1:8000/healthz`；若
Gateway 已退出，保留已有实验原始产物和进程日志，并在报告中明确缺少在线快照。指标和
报告不包含 prompt、API key 或 request ID label；分享前仍需按环境要求检查主机名和模型
标识。

## 2. 通用分诊

1. 记录告警开始时间、最近一次配置/模型/价格变更和受影响 tenant。
2. 检查 `/healthz`（进程）与 `/readyz`（执行器）；readiness 非 200 时立即从上游摘流量。
3. 导出现场包，保留 `requests.jsonl`，不要只截图仪表盘。
4. 先降低新流量或并发，再取消已确认无用的 batch 请求；不要删除原始记录。
5. 恢复后以同一固定 trace 做一次验证，确认资源、延迟、错误和成本回到已验收基线。

## 3. KV 压力

**判定信号**

- `cachepilot_reserved_kv_blocks / cachepilot_kv_capacity_blocks` 持续接近本机安全上限；
- `cachepilot_active_sequences` 接近 `cachepilot_active_sequence_capacity`；
- `cachepilot_admission_total{reason="kv_capacity"}`、`queue_full` 或 tenant 配额拒绝增速上升；
- long-context 请求增多，同时 queue P95/P99 和拒绝率恶化。

**止损与定位**

1. 暂停递增压测，限制新 long-context/batch 流量，不提高硬容量或降低 safety blocks。
2. 对照 `manifest.json` 的模型上下文、block size 与策略版本，确认不是配置漂移。
3. 从 `summary.json.resource_peaks.reserved_blocks_peak` 和逐请求 timeline 找到峰值请求；检查
   已终态请求是否仍占 reservation。
4. 若终态后占用不能回到空闲基线，按资源泄漏处理：摘流量、保存证据、重启前记录未释放
   request 状态。禁止用扩大上下文或重试风暴掩盖泄漏。

**恢复条件**：固定 long-context trace 可重复运行；每轮结束 reserved blocks 回到基线；无
超额接纳，拒绝原因与配置一致。

## 4. TTFT/TPOT 回退

先区分阶段，避免把排队问题误判为解码问题：

| 现象 | 优先检查 |
|---|---|
| queue 与 TTFT 同时上升，prefill 稳定 | 到达 burst、并发/KV 上限、调度策略 |
| prefill 与 TTFT 上升，queue 稳定 | prompt 长度、prefix 命中口径、模型/执行器变更 |
| TPOT/decode 上升，TTFT 基本稳定 | decode 批次、慢客户端背压、GPU 降频或争用 |
| TTFT、TPOT、错误同时上升 | worker 健康、OOM、驱动/执行器异常 |

1. 用同一 trace、seed、模型 revision 和执行器比较 `report.md`，模拟与实测禁止混比。
2. 确认已排除 warm-up；至少看 P50/P95/P99，不以单次请求定性。
3. 若只有个别 tenant 回退，检查 WFQ 权重、tenant 并发/token 配额和慢客户端。
4. 回滚最近的模型、量化、batch、admission 或 scheduler 变更时一次只改一个变量，并保存
   回滚前后两个 run。

**恢复条件**：至少一次固定 trace 重放回到既定 SLO/基线，吞吐、拒绝率和公平性没有以
不可接受幅度换取延迟改善。

## 5. Worker/执行器故障

**判定信号**：`/healthz` 正常但 `/readyz` 返回 503，或
`cachepilot_executor_healthy == 0`；同时 `cachepilot_errors_total` 的 executor/OOM 类错误
增加，请求终态出现 `FAILED`/`TIMED_OUT`。

1. 立即从上游摘流量，停止接纳新请求；不要在单机实例上无限重试。
2. 导出快照并记录 worker/驱动日志；核对失败请求最终只有一个终态，reservation 已释放。
3. OOM 时保留失败时的上下文、并发和 KV 峰值，降低并发或上下文后再重启；不得降低安全
   余量来复现“成功”。非 OOM 异常先验证模型文件、设备可见性和执行器版本。
4. 重启后先检查 `/readyz`，再用短 prompt 单并发探测，最后逐级恢复流量。

**恢复条件**：readiness 稳定为 200，executor healthy 为 1；故障注入或固定 trace 后所有
请求唯一终态，active sequences 与 reserved blocks 回到基线。

## 6. 成本突增

`cachepilot_estimated_gpu_seconds_total` 和 `cachepilot_estimated_cost_total` 都是进程生命周期
内单调累计 counter，必须比较等长时间窗口的增量，不能直接比较两个时刻的绝对总额。
成本按 `gpu_hour_price × (prefill_ms + decode_ms) / 1000 / 3600` 估算，不是云账单。

1. 先核对 `gpu_hour_price`、币种和策略版本是否变化；价格未配置时成本指标没有样本，但
   GPU seconds 仍可用于比较。
2. 将成本增量拆成请求量、prompt/completion token、prefill/decode 时间和失败/重试率；按
   tenant/model label 定位贡献者。
3. 检查是否出现重试风暴、超长输出、低命中 shared-prefix、慢客户端或 worker 故障。
4. 止损优先使用 tenant 并发/token/队列限制和输出上限；不要修改历史 ledger 或把估算值
   标成实际账单。

**恢复条件**：等长窗口的 GPU seconds 与估算成本增量回到预算基线；账本成本可由原始
prefill/decode 时间和价格重算；云账单差异单独记录。

## 7. 关闭事件

事件报告至少附带导出目录、配置/版本变更、影响窗口、止损动作、固定 trace 验证结果和
已知限制。只有当 readiness、资源回收、性能和成本四项均完成复核后才能恢复常态流量。
