# ADR-0003：模型与运行时依赖基线

- 状态：已接受
- 日期：2026-09-18
- 依赖：ADR-0001、ADR-0002

## 背景

模型仓库的默认分支、tokenizer、chat template 和 Python/GPU 软件栈都会随时间变化。即使 API、workload 和随机种子完全相同，使用浮动的 `main`、`latest`、版本范围或未记录的执行器镜像，也可能改变 token 数、输出、KV 容量估算和性能结果。

CachePilot 必须把模型与 tokenizer 当作实验输入的一部分，并区分三件事：

1. 上游包元数据声明彼此兼容；
2. CachePilot 选择并固定了某个组合；
3. 该组合已经在具体 GPU、驱动和 CUDA 环境中完成实测。

本 ADR 完成前两项。第三项必须在 Phase 1 环境验证中产生证据，不能仅凭版本声明为已验证。

## 模型决策

首版唯一模型固定为：

| 字段 | 固定值 |
|---|---|
| Model ID | `Qwen/Qwen2.5-0.5B-Instruct` |
| Model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Tokenizer ID | `Qwen/Qwen2.5-0.5B-Instruct` |
| Tokenizer revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| License | Apache-2.0 |
| Model maximum context | 32,768 tokens |
| Initial service limit | 8,192 tokens |
| Remote code | 禁止，`trust_remote_code=false` |

模型与 tokenizer 必须从同一个提交快照加载。不能只固定模型权重而让 tokenizer、chat template 或 generation config 从默认分支获取。

选择该模型的原因：

- 0.5B 参数规模适合单卡功能验证和受限环境 smoke test；
- 使用标准 Transformers `Qwen2ForCausalLM` 架构，不需要执行远程仓库代码；
- Apache-2.0 许可证允许本项目按当前目标使用和分发配置；
- 支持中英文聊天，适合作为项目开发和演示的共同基线；
- 具有 GQA 配置，可以用于验证 KV Planner 的 `num_key_value_heads` 口径；
- 32K 模型上下文足以覆盖后续长上下文 workload，但服务初期可以采用更保守的容量上限。

模型元数据记录在 `config/model.json`。首版架构字段为：

| 字段 | 值 |
|---|---:|
| hidden layers | 24 |
| attention heads | 14 |
| KV heads | 2 |
| hidden size | 896 |
| head dimension | 64 |

`head_dim` 必须等于 `hidden_size / num_attention_heads`。这些字段将作为后续 KV Planner 的输入，但本 ADR 不宣称它们等于执行器实际可用显存。

## 上下文限制

模型配置中的 `max_position_embeddings` 为 32,768。首版服务限制固定为 8,192，原因是：

- Phase 0 的重点是控制层不变量，不需要一开始覆盖模型理论最大窗口；
- 较小上限可以降低 CPU smoke test、TorchExecutor 和初期 GPU 实验的资源需求；
- 上限必须通过容量测试逐步提高，不能把单次成功视为安全容量结论。

请求必须同时满足：

```text
prompt_tokens + max_tokens <= initial_service_context_limit
initial_service_context_limit <= model_max_context_tokens
```

将服务限制提高到 32,768 以内不需要更换模型 revision，但需要配置变更、容量证据和回归测试。超过模型最大上下文或启用 RoPE scaling 需要新的 ADR。

## Python、PyTorch 与 vLLM 兼容矩阵

选定的 Python 版本为 `3.11.13`。所有正式开发、测试和实验环境必须使用 Python 3.11 系列中的这个精确版本，直到依赖升级 ADR 替换它。

| Profile | Platform | Python | PyTorch | Transformers | vLLM | 状态 |
|---|---|---:|---:|---:|---:|---|
| Phase 0 CPU | macOS/Linux CPU | 3.11.13 | 不安装 | 不安装 | 不安装 | 已选择；仓库骨架阶段验证 |
| TorchExecutor | Linux x86_64 + NVIDIA GPU | 3.11.13 | 2.11.0 | 5.5.3 | 不安装 | 元数据兼容；GPU 实测待完成 |
| VllmExecutor | Linux x86_64 + NVIDIA GPU | 3.11.13 | 2.11.0 | 5.5.3 | 0.24.0 | 元数据兼容；GPU 实测待完成 |

补充固定版本：

| 包 | 版本 |
|---|---:|
| tokenizers | 0.23.0 |
| safetensors | 0.6.2 |
| huggingface-hub | 1.5.0 |
| torchaudio | 2.11.0 |
| torchvision | 0.26.0 |

上游发布元数据确认：

- vLLM 0.24.0 支持 Python `>=3.10,<3.15`；
- vLLM 0.24.0 固定依赖 PyTorch 2.11.0；
- vLLM 0.24.0 要求 Transformers 5.5.3 或更高版本；
- Transformers 5.5.3 要求 tokenizers `>=0.22.0,<=0.23.0`。

CachePilot 不使用这些范围作为安装配置，而是在范围内选择精确版本。机器可读版本位于 `config/dependencies.json`，GPU 直接依赖约束位于 `requirements/constraints-gpu.txt`。

## CUDA 与平台边界

首版不在没有实际 GPU 环境的情况下固定一个未经验证的 CUDA/驱动组合。Phase 1 环境脚本必须记录并验证：

- GPU 型号和显存；
- NVIDIA driver；
- CUDA runtime 和 PyTorch CUDA build；
- PyTorch、Transformers 和 vLLM 实际导入版本；
- vLLM wheel 来源及其平台标签。

在这些证据产生前，兼容矩阵中的 GPU profile 只能标为“元数据兼容；GPU 实测待完成”。macOS 开发机只支持契约测试、Phase 0 模拟和不依赖 vLLM 的工作；不得把 macOS 环境列为 vLLMExecutor 支持目标。

## 安装与锁定策略

- `.python-version` 固定 Python 3.11.13；
- `config/model.json` 固定模型、tokenizer 和许可证来源；
- `config/dependencies.json` 是运行时组合的机器可读基线；
- `requirements/constraints-gpu.txt` 固定 GPU profile 的直接依赖；
- 模型加载必须显式传入 `revision` 和 `tokenizer_revision`；
- 模型 URL、许可证 URL 和配置 URL 必须包含 commit SHA，不能使用 `/main/`；
- 配置和安装命令中禁止 `latest`、`main`、`master`、`nightly`、裸版本范围和未标记容器镜像。

当前 constraints 文件固定直接依赖，不是完整的传递依赖 lock。仓库骨架任务需要选择锁文件工具，并针对 CPU 与 GPU 环境分别生成可复现 lock；生成 lock 前不得声称整个 Python 环境已经完全锁定。

## 升级规则

以下任一变更都必须通过新 ADR 或显式修订本 ADR：

- 模型或 tokenizer revision；
- chat template；
- 模型最大上下文或 RoPE 配置；
- Python minor/patch 基线；
- PyTorch、Transformers 或 vLLM 版本；
- dtype、量化方式或 `trust_remote_code`；
- GPU wheel/CUDA 基线。

升级必须同时更新机器可读配置、约束文件、环境记录和测试。禁止只修改文档表格或只修改安装命令。

## 验收证据

自动化测试需要证明：

1. model 和 tokenizer revision 都是 40 位 commit SHA；
2. 两者指向相同快照；
3. 许可证、上下文和架构字段存在；
4. service context 不超过模型最大上下文；
5. 所有选定包版本均为精确 `x.y.z`，没有版本范围；
6. `.python-version`、依赖配置和 constraints 一致；
7. OpenAPI 只接受固定的 model ID；
8. `main` 或 `latest` 等浮动值会使校验失败。

实现位置：

- `cachepilot/config/baseline.py`
- `config/model.json`
- `config/dependencies.json`
- `requirements/constraints-gpu.txt`
- `tests/contract/test_model_baseline.py`

## 来源

- 模型元数据：<https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct>
- 固定模型配置：<https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/config.json>
- 固定许可证：<https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE>
- vLLM 0.24.0 元数据：<https://pypi.org/pypi/vllm/0.24.0/json>
- Transformers 5.5.3 元数据：<https://pypi.org/pypi/transformers/5.5.3/json>

上述来源于 2026-09-18 核对。外部页面未来变化不改变仓库已经接受的固定版本。
