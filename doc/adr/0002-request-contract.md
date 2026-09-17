# ADR-0002：请求、流式响应与重试契约

- 状态：已接受
- 日期：2026-09-18
- 依赖：ADR-0001

## 背景

ADR-0001 固定了首版 API 的能力边界，但同一个合法聊天请求仍可能因 request ID、tenant、优先级、deadline、幂等键、token 统计或流式终止语义不同而产生不一致行为。这些差异会直接影响去重、调度、公平性、资源回收和实验结果。

本 ADR 固定首版请求的线协议。机器可读版本位于 `contracts/openapi.json`；不在该文件和本 ADR 中的行为不属于兼容承诺。

## 请求入口

首版聊天请求使用：

```text
POST /v1/chat/completions
Content-Type: application/json
Accept: text/event-stream
```

请求体继续采用 ADR-0001 的严格白名单：`model`、`messages`、`stream` 和可选的 `max_tokens`。`stream` 必须为 `true`。

控制信息只从 HTTP header 读取，不在 JSON 请求体中接受同名字段。

## Header 契约

### `X-Tenant-ID`

- 必填；是首版唯一 tenant 来源。
- 长度为 1–64 个字符。
- 首字符必须是 ASCII 字母或数字，其余字符只能是 ASCII 字母、数字、点、下划线或连字符。
- Gateway 不合并 body、query、cookie 或 OpenAI `user` 字段中的 tenant。
- 生产部署应由可信认证代理写入或覆盖该 header；首版 header 本身不构成认证机制。

缺失返回 `tenant_required`，格式错误返回 `invalid_tenant`。在 tenant 鉴权接入后，未知 tenant 可以返回 `tenant_not_authorized`，不得退化为匿名或默认 tenant。

### `X-Request-ID`

- 可选，由客户端提供时必须在整个 CachePilot 部署中唯一。
- 允许 1–128 个字符；首字符是 ASCII 字母或数字，其余字符只能是 ASCII 字母、数字、点、下划线、冒号或连字符。
- 缺失时服务端生成 `req_<uuid4 hex>`。
- 服务端通过响应 header `X-Request-ID`、SSE chunk 和错误体回传最终采用的 ID。
- 已存在的 request ID 不表示幂等重放；重复使用返回 HTTP 409 和 `request_id_conflict`。

日志、事件、取消和查询均使用该 ID。任何外部 request ID 都只能作为不透明标识，不得被解释成 tenant、时间或路由信息。

### `X-Priority`

- 可选，取值只能为 `interactive` 或 `batch`。
- 缺失时默认为 `interactive`。
- 它选择调度类别，不绕过 tenant quota、KV 上限或 deadline。
- 未知值返回 `invalid_priority`。

首版不接受任意数值优先级，避免客户端通过极大数值绕过公平策略。

### `X-Deadline-Ms`

- 可选，为十进制整数，范围 `1..3600000`。
- 表示从 Gateway 完成基本 HTTP 解析并接收请求时开始计算的总预算，覆盖 tokenization、排队、准入和执行。
- 服务内部必须用单调时钟保存 `received_monotonic + deadline_ms`，不能用墙上时钟判断超时。
- `interactive` 默认 30,000 ms；`batch` 默认 300,000 ms。
- 缺失时使用对应 priority 的默认值；无效值返回 `invalid_deadline`。

使用相对毫秒预算可以避免客户端与服务端时钟偏差。实验记录仍需另外保存 UTC 接收时间用于跨组件关联。

deadline 到期后，请求必须原子进入 `TIMED_OUT`，停止产生新 token，并最终释放 reservation。若 HTTP 响应尚未开始，返回 HTTP 504；若 SSE 已开始，则发送流内错误事件后结束流。

### `Idempotency-Key`

- 可选，长度为 1–128 个字符，格式与 `X-Request-ID` 相同。
- 作用域为 `(tenant_id, HTTP method, path, key)`。
- Gateway 对规范化请求计算 SHA-256 fingerprint。fingerprint 包括 tenant、priority、deadline 预算和完整规范化请求体，不包括 request ID 和 idempotency key 本身。
- 记录至少保留到原请求终态后的 24 小时；持久化和清理机制由 Registry 实现。

首版不缓存或重放完整 SSE，因此重复提交语义如下：

| 情况 | HTTP | 错误码 | 行为 |
|---|---:|---|---|
| key 首次出现 | 正常处理 | — | 创建一个请求 |
| 相同 key、相同 fingerprint、原请求未终态 | 409 | `idempotency_in_progress` | 返回原 request ID，不创建请求 |
| 相同 key、相同 fingerprint、原请求已终态 | 409 | `idempotency_replay_unavailable` | 返回原 request ID，不重放 SSE |
| 相同 key、不同 fingerprint | 409 | `idempotency_key_conflict` | 返回原 request ID，不创建请求 |

客户端可以使用 `GET /v1/requests/{request_id}` 查询原请求。未来如果实现安全的结果留存和重放，需要新 ADR 修改终态重复提交行为。

## 请求规范化

幂等 fingerprint 使用 UTF-8 编码的规范 JSON：对象 key 按字典序排序、无多余空白、保留消息顺序和字符串原值。省略的 `max_tokens` 先补为默认值 256，省略的 priority 和 deadline 也先补默认值，然后再计算 fingerprint。

以下差异会得到不同 fingerprint：

- tenant 不同；
- priority 或 deadline 不同；
- 模型、消息顺序、角色或文本不同；
- `max_tokens` 不同。

仅 request ID 不同不会改变 fingerprint。

## Token 口径

为了区分 API usage、KV 容量和执行器内部细节，固定以下术语：

- `prompt_tokens`：固定模型和 tokenizer revision 使用正式 chat template，并启用 generation prompt 后得到的输入 token ID 数量；包含模板插入的特殊 token。
- `completion_tokens`：实际作为文本增量交付给客户端的生成 token ID 数量。未交付的 EOS、停止 token、被取消后丢弃的 token 不计入该值。
- `total_tokens`：`prompt_tokens + completion_tokens`。
- `kv_tokens`：执行器实际保留 KV 的 token 数，仅用于内部容量和账本；它可以因特殊 token、预分配或执行器实现而与 API usage 不同，不能伪装成 `total_tokens`。
- `reserved_tokens`：准入策略为请求保留的逻辑容量，也不能作为实际 usage。

`max_tokens` 限制 `completion_tokens`，首版默认 256、协议硬上限 4096；部署还可以根据模型上下文设置更小的上限。达到上限时 `finish_reason` 为 `length`。

取消、超时和流内错误也必须记录截至终态前已交付的部分 usage。执行器无法提供可靠 token 口径时，字段应标记为不可用，不能用字符数或空格分词估算后冒充 tokenizer token。

## SSE 契约

成功开始流时返回：

```text
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
X-Request-ID: <request_id>
```

普通 chunk 使用未命名 SSE data 事件：

```text
data: {"id":"req_...","object":"chat.completion.chunk","model":"configured-model-id","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}

```

正常结束必须按顺序发送：

1. 一个终止 JSON chunk，`delta` 为空对象，`finish_reason` 为 `stop` 或 `length`，并携带最终 `usage`；
2. `data: [DONE]\n\n`；
3. 关闭响应流。

流开始后的取消、deadline 或执行错误使用具名错误事件：

```text
event: error
data: {"error":{"type":"request_error","code":"deadline_exceeded","message":"...","param":null,"request_id":"req_..."}}

data: [DONE]

```

一条流最多产生一个终止结果。发送终止 chunk 或 `event: error` 后不得再发送 token。只要连接仍可写，错误事件之后也发送 `[DONE]`。客户端已经断连时不再尝试网络写入，但内部仍必须完成唯一终态转换和资源回收。

首版不发送 heartbeat，不支持 `stream_options`，也不把取消表示成正常的 `finish_reason`。

## 错误响应

在 SSE 响应开始前，所有错误使用统一 JSON：

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "unknown_field",
    "message": "Unsupported request field: tools.",
    "param": "tools",
    "request_id": "req_..."
  }
}
```

稳定错误码和 HTTP 映射如下：

| HTTP | 错误码 |
|---:|---|
| 400 | `invalid_body`, `unknown_field`, `missing_field`, `invalid_type`, `invalid_messages`, `invalid_message`, `invalid_role`, `text_content_required`, `streaming_required`, `value_out_of_range`, `tenant_required`, `invalid_tenant`, `invalid_priority`, `invalid_deadline`, `invalid_request_id`, `invalid_idempotency_key` |
| 401/403 | `tenant_not_authorized` |
| 404 | `model_not_found`, `request_not_found` |
| 409 | `request_id_conflict`, `idempotency_in_progress`, `idempotency_replay_unavailable`, `idempotency_key_conflict`, `request_terminal` |
| 429 | `tenant_quota_exceeded`, `queue_full`, `admission_rejected` |
| 500 | `internal_error`, `executor_failed` |
| 503 | `service_unavailable`, `worker_unavailable` |
| 504 | `deadline_exceeded` |

错误 `type` 是面向兼容层的粗分类，程序逻辑必须读取稳定的 `code`。错误信息可以改善措辞，但不能作为机器判断依据。

## 查询与取消

请求查询使用：

```text
GET /v1/requests/{request_id}
X-Tenant-ID: <tenant>
```

取消使用：

```text
POST /v1/requests/{request_id}/cancel
X-Tenant-ID: <tenant>
```

规则如下：

- tenant 只能查询或取消自己的请求；不存在和属于其他 tenant 的请求均返回 `request_not_found`，避免泄漏 ID 是否存在。
- 第一次取消非终态请求返回 HTTP 202；Registry 负责竞争并产生唯一终态。
- 重复取消已经 `CANCELLED` 的请求返回 HTTP 200 和相同状态，不重复释放资源。
- 取消已经 `FINISHED`、`TIMED_OUT`、`REJECTED` 或 `FAILED` 的请求返回 HTTP 409 和 `request_terminal`。
- 客户端断开 SSE 连接等价于发起取消，但网络断开本身不能覆盖已经提交成功的其他终态。
- 一旦 `CANCELLED` 成为终态，不得再输出 token，所有逻辑 reservation 必须最终释放一次。

## 重试语义

- 400、401、403、404 和不可恢复的 409 不应原样重试。
- `idempotency_in_progress` 应查询原 request ID，而不是创建新请求。
- 429、503 和在响应开始前发生的 504 可以按照 `Retry-After` 重试，但应复用原 `Idempotency-Key`。
- 客户端已经收到任何 SSE token 后，不得盲目自动重试生成请求；应先查询原 request ID，避免重复生成和计费。
- 收到 `[DONE]` 表示本次流协议已经结束，不应重试。
- 未提供 `Idempotency-Key` 的重试被视为全新请求，服务不保证去重。

服务端只在建议重试的响应中发送 `Retry-After`。该 header 的值使用整数秒。

## 可执行契约

- `cachepilot/gateway/contracts.py` 实现严格请求验证、规范化 fingerprint 和最小原子幂等 guard。
- `contracts/openapi.json` 固定 HTTP schema、header、响应和 SSE 扩展说明。
- `tests/contract/test_request_contract.py` 覆盖合法、非法和重复提交。

当前幂等 guard 只用于锁定外部行为，不替代后续 Request Registry。Registry 接入时必须保持本 ADR 的观察结果，并补充并发、持久化和保留期测试。

## 后果

- 客户端不需要同步时钟即可声明 deadline。
- request ID 与幂等键职责分离：前者用于追踪，后者用于去重。
- 首版不会重放已完成请求的 SSE，客户端必须通过查询接口了解原终态。
- API usage 与 KV/reservation 统计被明确分离，实验不能混用指标。
- 未来加入非流式响应、结果重放或更多优先级需要显式修改契约。
