# ADR-0001：首版 API 与能力范围

- 状态：已接受
- 日期：2026-09-18
- 决策范围：CachePilot 首个可运行版本

## 背景

CachePilot 对外提供 OpenAI 风格的聊天生成接口，但首版不能把“请求格式相似”表述为完整的 OpenAI API 兼容。执行器、请求生命周期、租户隔离和资源回收都需要可验证的语义；如果 API 静默忽略尚未实现的字段，客户端会误以为工具调用、多模态输入、采样参数或结构化输出已经生效，实验结果也会失去可解释性。

因此，首版采用最小能力集合和严格字段白名单。只有在实现、文档和契约测试同时具备后，能力才可以加入支持列表。

## 决策

首版只支持以下能力：

1. 单个固定模型。服务启动配置确定唯一的模型 ID 和 revision；请求不能动态选择其他模型。
2. 每个请求恰好归属一个 tenant。tenant 的具体来源、冲突处理和认证方式由后续请求契约 ADR 定义。
3. 仅提供聊天生成接口 `POST /v1/chat/completions`。
4. 消息角色仅支持 `system`、`user` 和 `assistant`。
5. 消息内容仅支持纯文本字符串。
6. 每个请求只生成一个结果，等价于固定 `n=1`。
7. 首个端到端输出路径仅支持 `stream=true`，通过 Server-Sent Events（SSE）返回。
8. 请求体首版只接受 `model`、`messages`、`stream` 和 `max_tokens`。
9. Prefix/KV 元数据强制按 tenant 隔离，不允许跨 tenant 查询、命中或共享。

`max_tokens` 可以省略，由服务端使用固定且有界的默认值。其取值上限、与上下文窗口的关系以及 token 统计口径由后续请求契约定义。

最小合法请求示例：

```json
{
  "model": "configured-model-id",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "stream": true,
  "max_tokens": 128
}
```

这里的“OpenAI 风格”只表示接口路径、消息基本结构和 SSE 数据形态与 OpenAI Chat Completions 接近，不表示完整兼容 OpenAI API。

## 严格验证策略

请求 schema 采用允许列表，而不是忽略未知字段：

- 请求体或消息对象出现未知字段时拒绝请求；
- 已知但尚未支持的字段即使值为 `null`、空数组或默认值也拒绝请求；
- `model` 与服务配置的唯一模型不一致时拒绝请求；
- `stream` 缺失或不为 `true` 时拒绝请求；
- `messages` 为空、角色无效或 `content` 不是字符串时拒绝请求；
- 不对字符串、布尔值和整数进行可能改变语义的宽松类型转换；
- 验证失败的请求不得进入 Registry、Admission、Scheduler 或占用 KV reservation。

具体 HTTP 状态码、错误响应结构和参数定位格式由后续请求契约统一定义。在该契约完成前，任何实现都不得通过丢弃字段来使请求成功。

建议使用 Pydantic v2 的严格模型实现该策略：

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: Literal[True]
    max_tokens: int = Field(default=256, ge=1)
```

模型 ID 需要额外与运行时配置比较，不能只验证其类型。

## 明确不支持

所有未出现在首版请求字段允许列表中的 OpenAI 字段默认不支持。以下类别需要特别明确地拒绝，不能静默忽略。

### 工具与函数调用

- `tools`
- `tool_choice`
- `functions`
- `function_call`
- `parallel_tool_calls`

### 多模态输入与输出

- 图片、音频、视频和文件内容块
- URL 或上传文件引用
- `modalities`
- `audio`

`messages[*].content` 的数组形式也不支持，即使数组中只有文本内容块。

### 多结果、高级生成与响应控制

- `n`
- `temperature`
- `top_p`
- `stop`
- `seed`
- `logprobs`
- `top_logprobs`
- `logit_bias`
- `response_format`
- `stream_options`
- `prediction`
- `reasoning_effort`

这些字段需要等执行器间语义可以对齐并具有契约测试后，才能通过新的 ADR 加入支持范围。

### 平台与持久化字段

- `user`
- `service_tier`
- `store`
- `metadata`

OpenAI 的 `user` 字段不能作为 CachePilot tenant 身份的替代来源。

### 其他 API 与运行模式

- 非流式聊天响应
- Completions API
- Responses API
- Embeddings API
- 图像或音频生成 API
- 多模型选择与路由
- 单请求多 tenant
- 跨 tenant Prefix/KV 共享

后续 Phase 1 可以通过更新本 ADR 或新增 ADR 增加非流式聊天，但在对应 schema、实现和契约测试完成前不得宣称支持。

## Tenant 与 Prefix/KV 隔离

所有可能影响缓存复用的逻辑 key 至少包含以下隔离维度：

```text
tenant_id
model_id
model_revision
tokenizer_revision
quantization_config
tokenized_prefix
```

相同文本在不同 tenant 下必须产生不同的逻辑缓存作用域。首版不提供关闭 tenant 隔离的配置项，也不允许为了提高命中率回退到全局 prefix 查询。

逻辑 prefix 命中不代表执行器发生了物理 KV 复用。两者必须分别记录，执行器没有提供可靠信号时，物理命中应标记为“不可观测”。

## 验收证据

完成请求 schema 和 API 骨架后，至少需要以下自动化契约测试：

1. 最小合法请求通过验证。
2. 未知顶层字段（例如拼错的 `max_token`）被拒绝。
3. 消息对象中的未知字段被拒绝。
4. `tools=null` 和 `tools=[]` 均被拒绝。
5. 图片或内容块数组被拒绝。
6. `stream=false` 被拒绝。
7. 与配置不一致的模型 ID 被拒绝。
8. `n=2` 被拒绝。
9. 错误类型和空消息列表被拒绝。
10. 验证失败的请求不进入 Registry，且不产生 KV reservation。
11. 两个 tenant 使用相同 tokenized prefix 时不会发生跨 tenant 逻辑命中。

在 ADR 存在但上述可执行证据尚未完成时，`TODO.md` 中的“确定首版范围”任务仍保持未完成状态。

## 后果

正面影响：

- 客户端可以明确知道哪些参数真正生效；
- Sim、Torch 和 vLLM 执行器可以围绕同一最小语义实现；
- 实验不会混入被忽略字段造成的不可控差异；
- 后续扩展能力时可以通过 ADR、schema 和契约测试审查兼容性。

代价与限制：

- 首版不是完整的 OpenAI API 替代品；
- 常见采样字段和非流式响应暂时不可用；
- 已经使用宽松 OpenAI 请求体的客户端需要先裁剪字段；
- 每次增加字段都必须同时处理不同执行器的语义和测试。

## 后续工作

下一份请求契约应继续确定：

- request ID 的生成与透传；
- tenant 的唯一来源和冲突规则；
- priority、deadline 和 idempotency key；
- `max_tokens` 默认值、边界和 token 口径；
- SSE chunk 与结束事件；
- 错误码和错误响应结构；
- 取消、客户端断连与重试语义。
