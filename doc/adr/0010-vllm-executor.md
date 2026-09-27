# ADR-0010：VllmExecutor 只适配公开异步引擎边界

## 状态

Accepted

## 决策

`VllmExecutor` 使用锁定版本 vLLM 的 `AsyncLLMEngine`、`AsyncEngineArgs`、
`SamplingParams` 和 `RequestOutputKind.DELTA`。CachePilot 只负责：

- 用固定 tokenizer/chat template 生成 prompt 和准入 token 数；
- 把外部 request ID 原样传给 `generate`，并校验每个输出的 request ID；
- 消费 delta text/token IDs，保存 prompt、completion 和 total token usage；
- 把断连、显式取消和超时映射为同一 request ID 的 `abort`；
- 把引擎异常转换为稳定的 `VllmExecutorError`。

continuous batching、调度、PagedAttention、物理 KV 和 GPU kernel 均由 vLLM
拥有。CachePilot 不增加第二层执行 batch，也不从 vLLM 私有对象推断物理 KV。

## 验证边界

CPU 测试通过注入 fake async engine 核对协议，不要求安装 vLLM。真实单 GPU环境
必须安装 `config/dependencies.json` 锁定的 vLLM 版本；版本不一致时适配器拒绝启动。
