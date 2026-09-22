# CachePilot 项目结构与文件功能说明

本文档按当前 Git 版本库中的文件整理，说明项目定位、目录结构以及每个文件的作用。
`__pycache__`、`.DS_Store`、`.idea/workspace.xml` 等本地生成或已忽略文件不属于项目交付内容，
因此不逐项列出。

## 1. 项目概览

CachePilot 是一个面向多租户混合负载的 LLM 推理服务控制层，计划构建在 vLLM、
PyTorch 等推理执行器之上。它不重新实现 CUDA kernel 或模型内部的连续批处理，
主要负责请求校验、生命周期管理、KV 容量准入、公平调度、多 GPU 路由、故障恢复、
遥测和实验分析。

当前仓库仍处于骨架阶段，已经具备以下可执行能力：

- 严格校验首版聊天补全请求，并规范化租户、优先级、截止时间等请求信息；
- 防止请求 ID 或幂等键导致的重复提交；
- 通过租户作用域的精确前缀键验证逻辑缓存隔离；
- 校验模型、Python 和 GPU 软件栈的固定版本基线；
- 校验实验产物并计算延迟分位数、吞吐量、拒绝率和公平性等摘要；
- 启动只提供健康检查、就绪检查和指标端点的空 HTTP 服务；
- 通过 CPU/Colab 流程运行静态检查、测试和服务探活。

真实模型执行器、调度循环和多 GPU Worker 路由尚未实现，
属于 `doc/TODO.md` 中规划的 Phase 0–2 工作。

## 2. 文件结构

```text
CachePilot/
├── .gitignore                         # Git 忽略规则
├── .python-version                    # 固定 Python 3.13.15 版本
├── Makefile                           # 安装、检查、测试和服务命令入口
├── README.md                          # 项目概览、目标和使用说明
├── pyproject.toml                     # Python 包、依赖和工具配置
├── .idea/
│   ├── .gitignore                     # 忽略 IDE 本地状态
│   ├── cachepilot.iml                 # PyCharm Python 模块配置
│   ├── modules.xml                    # PyCharm 模块注册
│   ├── vcs.xml                        # PyCharm Git 映射配置
│   └── inspectionProfiles/
│       └── profiles_settings.xml      # IDE 代码检查配置
├── cachepilot/
│   ├── __init__.py                    # CachePilot 控制平面包入口
│   ├── __main__.py                    # python -m cachepilot 启动入口
│   ├── cache/
│   │   ├── __init__.py                # 缓存元数据包入口
│   │   └── prefix_index.py            # 租户隔离的逻辑前缀索引
│   ├── config/
│   │   ├── __init__.py                # 配置校验包入口
│   │   └── baseline.py                # 模型和依赖版本基线校验
│   ├── executors/
│   │   ├── __init__.py                # 执行器公共接口
│   │   └── sim.py                     # 逻辑时钟 continuous-batching 模拟器
│   ├── gateway/
│   │   ├── __init__.py                # 网关包入口
│   │   ├── contracts.py               # 请求契约、规范化和幂等校验
│   │   └── intake.py                  # 校验通过后的请求接收边界
│   ├── routing/
│   │   └── __init__.py                # 多 Worker 路由包占位
│   ├── runtime/
│   │   ├── __init__.py                # 运行时包入口并导出服务创建函数
│   │   ├── adaptive_admission.py      # P95 自适应 KV 准入与增长保护
│   │   ├── admission.py               # Strict KV 准入与有界排队
│   │   ├── deadlines.py               # 单调时钟排队/执行超时与资源回收
│   │   ├── empty_service.py           # 健康检查、就绪和指标空服务
│   │   ├── registry.py                # 请求查询、去重、状态和事件日志 Registry
│   │   ├── resources.py               # 逻辑 KV 租约和物理 handle 资源账本
│   │   ├── kv_planner.py              # KV block 与理论字节容量规划
│   │   ├── scheduler.py               # FCFS/WFQ 优先级与 tenant 公平调度
│   │   └── state_machine.py           # 请求生命周期状态机和 token 输出门禁
│   └── telemetry/
│       └── __init__.py                # 指标、trace 和成本账本包占位
├── config/
│   ├── dependencies.json              # CPU/GPU 依赖版本和兼容性基线
│   ├── experiment.schema.json         # 实验 trace、记录和汇总 JSON Schema
│   └── model.json                     # 模型、分词器、架构和上下文配置
├── contracts/
│   └── openapi.json                   # 首版 HTTP API OpenAPI 契约
├── benchmarks/
│   ├── README.md                      # 基准实验产物和分析命令说明
│   ├── analyze.py                     # 实验数据校验和统计摘要生成
│   └── colab_acceptance.py            # CPU/Colab 骨架验收脚本
├── workloads/
│   ├── README.md                      # 固定工作负载和 trace 规范说明
│   └── smoke.jsonl                    # 最小单请求 trace 样例
├── requirements/
│   ├── constraints-cpu.txt            # CPU 测试工具固定版本
│   ├── constraints-gpu.txt            # PyTorch/vLLM 等 GPU 依赖版本
│   └── requirements-cpu.txt           # CPU 环境可复现安装入口
├── tests/
│   ├── __init__.py                    # 测试包入口
│   ├── unit/
│   │   ├── __init__.py                # 单元测试包入口
│   │   ├── test_empty_service.py      # 空服务端点测试
│   │   ├── test_registry.py           # Registry 查询、幂等和并发终态测试
│   │   ├── test_resources.py          # 资源申请、增长、释放和回收测试
│   │   ├── test_scheduler.py          # FCFS/WFQ 顺序、公平与重放测试
│   │   ├── test_sim_executor.py       # 模拟执行、背压、取消和故障测试
│   │   └── test_state_machine.py      # 生命周期状态转换与幂等测试
│   ├── integration/
│   │   ├── __init__.py                # 集成测试包入口
│   │   └── test_imports.py            # 各子包导入冒烟测试
│   ├── contract/
│   │   ├── __init__.py                # 契约测试包入口
│   │   ├── test_experiment_protocol.py # 实验协议校验测试
│   │   ├── test_model_baseline.py     # 模型和依赖基线测试
│   │   └── test_request_contract.py   # 请求、租户和幂等契约测试
│   ├── load/
│   │   └── README.md                  # 负载测试目录说明和预留范围
│   └── chaos/
│       └── README.md                  # 故障注入测试目录说明和预留范围
├── doc/
│   ├── DESIGN.md                      # 总体架构和工程设计
│   ├── TODO.md                        # 分阶段实施任务和资源规划
│   ├── PROJECT_STRUCTURE.md           # 本文件：文件树和功能说明
│   ├── acceptance/
│   │   └── 0004-repository-skeleton.md # 仓库骨架验收记录
│   └── adr/
│       ├── 0001-initial-api-scope.md   # 首版 API 范围决策
│       ├── 0002-request-contract.md    # 请求、SSE 和重试契约决策
│       ├── 0003-model-and-dependency-baseline.md # 模型和依赖基线决策
│       ├── 0004-python-3.13-baseline.md # Python 版本基线决策
│       └── 0005-experiment-protocol.md # 可复现实验协议决策
├── deploy/
│   └── README.md                      # 部署配置现状和后续规划
└── notebooks/
    └── colab_acceptance.ipynb         # Google Colab 骨架验收流程
```

## 3. 根目录文件

| 文件 | 功能 |
|---|---|
| `.gitignore` | 定义 Git 忽略规则，排除系统文件、Python 缓存、虚拟环境、构建产物、测试缓存、日志、临时文件、密钥配置和 IDE 本地状态。 |
| `.python-version` | 将项目 Python 基线固定为 `3.13.15`，供 pyenv 等版本管理工具和契约测试读取。 |
| `Makefile` | 提供常用开发命令：CPU 依赖安装、Ruff 检查、pytest 测试、综合检查、空服务启动和 Colab 验收。 |
| `README.md` | 项目首页，概述目标、架构边界、Phase 0–2 路线、完成标准、安装方式和主要文档入口。 |
| `pyproject.toml` | Python 包构建与项目元数据配置；声明 Python 版本、开发依赖、命令行入口、包发现方式，以及 pytest 和 Ruff 配置。 |

## 4. IDE 配置：`.idea/`

这些文件用于 JetBrains/PyCharm 项目识别，不参与 CachePilot 的运行时逻辑。

| 文件 | 功能 |
|---|---|
| `.idea/.gitignore` | 忽略 PyCharm 的 shelf、工作区状态和本地 HTTP 请求记录。 |
| `.idea/cachepilot.iml` | 定义 PyCharm Python 模块、内容根目录和项目 SDK 信息。 |
| `.idea/modules.xml` | 将 `cachepilot.iml` 注册为当前 IDE 项目的模块。 |
| `.idea/vcs.xml` | 声明项目根目录使用 Git 版本控制。 |
| `.idea/inspectionProfiles/profiles_settings.xml` | 配置 IDE 代码检查配置文件的选择方式。 |

## 5. 主程序包：`cachepilot/`

### 5.1 包入口

| 文件 | 功能 |
|---|---|
| `cachepilot/__init__.py` | 标记 `cachepilot` 为 Python 包，并说明它是 CachePilot 控制平面包。 |
| `cachepilot/__main__.py` | 支持执行 `python -m cachepilot`，实际转交给空服务的 `main()` 函数。 |

### 5.2 缓存元数据：`cachepilot/cache/`

| 文件 | 功能 |
|---|---|
| `cachepilot/cache/__init__.py` | 声明租户作用域缓存元数据包。 |
| `cachepilot/cache/prefix_index.py` | 定义包含租户、模型版本、分词器版本、量化配置和 token 前缀的不可变缓存键，并提供租户隔离的精确逻辑前缀索引。目前只支持记录和精确匹配，不支持最长前缀、淘汰或物理 KV 命中。 |

### 5.3 配置校验：`cachepilot/config/`

| 文件 | 功能 |
|---|---|
| `cachepilot/config/__init__.py` | 声明配置加载和基线校验包。 |
| `cachepilot/config/baseline.py` | 将模型和依赖 JSON 加载为不可变数据对象；检查模型/分词器 Git SHA、架构参数、上下文长度、精确依赖版本和禁用浮动版本，并核对 `.python-version` 与 GPU constraints 是否一致。 |

### 5.4 执行器边界：`cachepilot/executors/`

| 文件 | 功能 |
|---|---|
| `cachepilot/executors/__init__.py` | 导出 SimExecutor 的逻辑时钟、请求、事件、快照、统计和稳定异常；Torch/vLLM 适配器仍待 Phase 1 实现。 |
| `cachepilot/executors/sim.py` | 实现固定 tick 的确定性模拟执行器，覆盖 prefill/decode、逻辑 KV 增长和释放、continuous batching、客户端背压、取消、worker 故障及事件/统计快照。 |

### 5.5 网关：`cachepilot/gateway/`

| 文件 | 功能 |
|---|---|
| `cachepilot/gateway/__init__.py` | 声明网关契约及 HTTP 辅助组件包。 |
| `cachepilot/gateway/contracts.py` | 实现首版聊天补全请求的严格校验与规范化，定义稳定错误结构、请求指纹、请求 ID/幂等键规则，以及线程安全的重复提交防护。首版仅接受纯文本消息和 `stream=true`。 |
| `cachepilot/gateway/intake.py` | 定义请求进入运行时前的校验边界。只有通过契约校验并完成规范化的请求才会交给下游 `AcceptedRequestSink`，避免无效输入创建生命周期状态或 KV 预留。 |

### 5.6 路由：`cachepilot/routing/`

| 文件 | 功能 |
|---|---|
| `cachepilot/routing/__init__.py` | 预留请求路由包；计划在 Phase 2 接入 Worker Directory、最少负载和缓存感知路由。 |

### 5.7 运行时：`cachepilot/runtime/`

| 文件 | 功能 |
|---|---|
| `cachepilot/runtime/__init__.py` | 声明运行时包，导出请求状态机、准入、deadline、资源账本、FCFS/WFQ 调度器和 `create_server`；后续计划继续承载 Worker loop。 |
| `cachepilot/runtime/adaptive_admission.py` | 按 tenant 和 prompt 长度分桶统计历史输出 P95，加安全余量进行自适应 reservation；样本不足回退 Strict，并在生成增长时阻止突破 KV 或 tenant 硬上限。 |
| `cachepilot/runtime/admission.py` | 实现线程安全的 Strict Admission：按 prompt 与最大输出长度预留 KV blocks，并原子限制全局活跃数、安全容量、tenant token/并发及全局/tenant 队列长度；返回稳定的接纳、排队或拒绝原因。 |
| `cachepilot/runtime/deadlines.py` | 使用单调时钟跟踪请求总 deadline、排队 deadline 和执行 deadline；由 runtime loop 确定性扫描到期请求并调用资源回收回调。 |
| `cachepilot/runtime/empty_service.py` | 使用 Python 标准库实现线程化空 HTTP 服务，提供 `/healthz`、`/readyz`、`/metrics` 和统一 404 响应。它只用于环境验收，不执行模型推理，也不实现正式 OpenAI API。 |
| `cachepilot/runtime/registry.py` | 实现线程安全的内存请求 Registry，按 request ID 和租户作用域幂等键注册、查询和去重，并通过不可变快照暴露当前状态、token 数与状态事件日志。 |
| `cachepilot/runtime/resources.py` | 实现线程安全的资源租约账本：管理逻辑 KV block 的申请、增长、容量和一次性释放，并独立记录执行器物理 handle。 |
| `cachepilot/runtime/scheduler.py` | 实现线程安全且可注入逻辑时钟的 FCFS/WFQ 调度：维护 interactive/batch 下的 tenant FIFO 子队列，以精确分数计算 WFQ 虚拟标签，并按最大队首等待时间提供饥饿保护。 |
| `cachepilot/runtime/kv_planner.py` | 根据模型架构、KV dtype、block size 和服务上下文限制计算请求逻辑 blocks 与理论 KV 字节；只有显式给出 KV 专用字节预算时才换算容量，不从 GPU 总显存推导真实可用容量。 |
| `cachepilot/runtime/state_machine.py` | 实现单请求生命周期状态机、原子状态迁移、事件 ID 幂等与冲突检测、不可逆终态，以及仅在 `EXECUTING` 状态开放的 token 输出登记门禁。 |

### 5.8 遥测：`cachepilot/telemetry/`

| 文件 | 功能 |
|---|---|
| `cachepilot/telemetry/__init__.py` | 预留指标、trace 和成本账本的包边界，当前没有具体采集实现。 |

## 6. 机器可读配置：`config/`

| 文件 | 功能 |
|---|---|
| `config/dependencies.json` | 固定 Phase 0 CPU、Torch 执行器和 vLLM 执行器的软件栈版本，记录平台条件、兼容性状态、上游约束和来源。 |
| `config/experiment.schema.json` | 实验协议的 JSON Schema，定义 trace、运行 manifest、逐请求记录和汇总结果四类对象的字段、类型和约束。 |
| `config/model.json` | 固定初始模型 `Qwen/Qwen2.5-0.5B-Instruct`、模型与分词器 revision、许可证、架构、上下文上限、数据类型和元数据来源。 |

## 7. API 契约：`contracts/`

| 文件 | 功能 |
|---|---|
| `contracts/openapi.json` | 首版 API 的 OpenAPI 3.1 契约，描述流式聊天补全、请求状态查询和取消端点，以及控制请求头、严格请求对象、SSE 响应和错误响应。该文件表达目标接口，不代表所有端点已有 HTTP 实现。 |

## 8. 基准与验收：`benchmarks/`

| 文件 | 功能 |
|---|---|
| `benchmarks/README.md` | 说明一次实验必须保存的 manifest、trace、逐请求记录和 summary，并给出分析器命令及有效性边界。 |
| `benchmarks/analyze.py` | 校验实验 manifest、trace 和逐请求 JSONL；检查版本、时钟、字段、终态和请求 ID 一致性；生成延迟分位数、吞吐量、拒绝率、取消率和 Jain 公平性摘要。 |
| `benchmarks/colab_acceptance.py` | CPU/Colab 骨架验收脚本；检查 Python `3.13.15`，运行 pytest，启动空服务并探测三个健康端点，成功时输出 `COLAB_ACCEPTANCE=PASS`。 |

## 9. 工作负载：`workloads/`

| 文件 | 功能 |
|---|---|
| `workloads/README.md` | 说明固定 trace 的字段、随机种子和可重放要求，并强调当前样例不是性能数据。 |
| `workloads/smoke.jsonl` | 最小单请求 trace 样例，用于验证文件路径和实验协议格式。 |

## 10. 依赖文件：`requirements/`

| 文件 | 功能 |
|---|---|
| `requirements/constraints-cpu.txt` | 固定 CPU 开发和测试使用的 pytest、Ruff 版本。 |
| `requirements/constraints-gpu.txt` | 固定计划中的 GPU 直接依赖版本，包括 PyTorch、Transformers 和 vLLM；它是约束集合，不是完整 lock 文件。 |
| `requirements/requirements-cpu.txt` | CPU 环境的可复现安装入口，引用 CPU constraints，以 editable 模式安装项目并安装测试、检查工具。 |

## 11. 测试：`tests/`

| 文件 | 功能 |
|---|---|
| `tests/__init__.py` | 标记顶层测试包。 |
| `tests/unit/__init__.py` | 标记 CPU 单元测试包。 |
| `tests/unit/test_empty_service.py` | 启动临时空服务，验证健康、就绪和 Prometheus 文本指标端点均能正确响应。 |
| `tests/unit/test_registry.py` | 验证请求注册与查询、request ID/幂等键去重、租户隔离、状态快照和取消/完成/失败并发竞争只产生一个终态。 |
| `tests/unit/test_resources.py` | 验证 reservation 申请、增长、容量限制、物理 handle 隔离、一次性释放和所有终态路径回收到基线。 |
| `tests/unit/test_state_machine.py` | 验证主路径、非法转换、重复事件、事件冲突、任意非终态进入异常终态、终态不可逆和终态后禁止输出 token。 |
| `tests/unit/test_scheduler.py` | 验证 FCFS 优先级与类内顺序、tenant FIFO 子队列、WFQ 权重和虚拟完成标签、最大饥饿提升及固定 trace 确定性重放。 |
| `tests/unit/test_sim_executor.py` | 验证逻辑时钟推进、prefill/decode、KV block 增长、continuous batch 补位、慢客户端背压、取消、worker 故障和相同输入完全一致重放。 |
| `tests/integration/__init__.py` | 标记集成冒烟测试包。 |
| `tests/integration/test_imports.py` | 验证 cache、config、executors、gateway、routing、runtime 和 telemetry 等包都可以成功导入。 |
| `tests/contract/__init__.py` | 标记可执行契约测试包。 |
| `tests/contract/test_experiment_protocol.py` | 验证实验 Schema 是合法 JSON、缺失关键元数据会失败，以及分析器可从协议记录生成摘要。 |
| `tests/contract/test_model_baseline.py` | 验证模型/分词器 revision、许可证、架构、上下文和依赖版本均被固定，并检查 OpenAPI 只接受选定模型。 |
| `tests/contract/test_request_contract.py` | 全面验证请求字段、消息类型、请求头、默认值、请求指纹、校验优先边界、租户缓存隔离、幂等语义和 OpenAPI 引用。 |
| `tests/load/README.md` | 为 Phase 0/1 固定 trace 回放与受控压力测试预留目录。 |
| `tests/chaos/README.md` | 为取消、超时、Worker 故障和恢复测试预留目录。 |

## 12. 项目文档：`doc/`

| 文件 | 功能 |
|---|---|
| `doc/DESIGN.md` | 总体工程设计，覆盖目标函数、阶段范围、架构、API、状态机、单/多 GPU Runtime、KV 与调度、可观测性、实验方法、安全和验收标准。 |
| `doc/TODO.md` | 按依赖顺序列出 Phase 0–2 的实施任务、预计工时、硬件资源、风险和功能缩减顺序。 |
| `doc/PROJECT_STRUCTURE.md` | 本文档；集中展示仓库文件树，并用中文解释每个版本控制文件的职责。 |
| `doc/acceptance/0004-repository-skeleton.md` | 仓库骨架的验收记录，列出交付内容、CPU/Colab 命令、已验证证据和当前限制。 |
| `doc/adr/0001-initial-api-scope.md` | 决定首版 API 支持范围、严格字段策略、不支持能力，以及 Tenant 与 Prefix/KV 隔离规则。 |
| `doc/adr/0002-request-contract.md` | 决定请求入口、控制 Header、规范化、token 口径、SSE、错误、查询/取消和重试语义。 |
| `doc/adr/0003-model-and-dependency-baseline.md` | 记录初始模型选择、上下文限制、Python/PyTorch/vLLM 兼容矩阵、CUDA 边界、锁定和升级规则。 |
| `doc/adr/0004-python-3.13-baseline.md` | 记录采用 Python `3.13.15` 作为统一实验基线的原因、验收要求和影响。 |
| `doc/adr/0005-experiment-protocol.md` | 决定可复现实验的运行单位、trace schema、时钟、seed、manifest、逐请求记录、汇总和验收规则。 |
| `doc/adr/0006-fcfs-and-wfq-scheduling.md` | 决定 FCFS/WFQ 的优先级、tenant 子队列、虚拟时间、权重、稳定打破平局和最大饥饿时间语义。 |
| `doc/adr/0007-sim-executor.md` | 决定 SimExecutor 的逻辑 tick 顺序、prefill/decode 成本、KV 生命周期、输出缓冲、取消/故障和确定性重放语义。 |

## 13. 部署与 Notebook

| 文件 | 功能 |
|---|---|
| `deploy/README.md` | 说明当前阶段暂不提供生产部署配置，待 API 和运行时稳定后再补充单进程服务定义。 |
| `notebooks/colab_acceptance.ipynb` | Google Colab 骨架验收笔记本，按顺序完成工作目录准备、CPU 依赖安装、`make check` 和空服务验收。 |

## 14. 阅读顺序建议

第一次了解项目时，建议按以下顺序阅读：

1. `README.md`：快速理解项目目标、边界和当前状态；
2. `doc/DESIGN.md`：理解目标架构和阶段设计；
3. `doc/TODO.md`：了解尚未实现的能力和开发顺序；
4. `contracts/openapi.json` 与 `doc/adr/0001-0002`：理解 API 契约；
5. `cachepilot/gateway/`、`cachepilot/config/`、`cachepilot/cache/`：阅读当前核心实现；
6. `tests/contract/`：通过可执行测试确认契约的精确边界；
7. `doc/adr/0005-experiment-protocol.md` 与 `benchmarks/analyze.py`：理解实验数据规范。
