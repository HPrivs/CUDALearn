# CUDA 算子渐进优化学习项目

本项目目标不是一次写出“最快代码”，而是按顺序学会：

1. 写出正确 CUDA kernel。
2. 用可复现 benchmark 衡量性能。
3. 用访存量、FLOPs、算术强度和 profiler 指标解释瓶颈。
4. 每轮只引入一个主要优化手段，并说清它为什么有效、何时失效。

讲解默认使用中文；CUDA API、kernel 名、metric 名和命令保持英文。

## 项目结构

```text
root/
├── AGENTS.md
├── src/          # common.cuh, X.cu
├── doc/          # _roadmap.md, X.md, _nsight.md, _glossary.md
├── homework/     # X.md，用户作业和批改反馈
└── debugger/     # 编译输出
```

## 请求分类

先判断是否是项目维护请求：

- 维护请求：明确要求修改 `AGENTS.md`、路线图、目录规则、文档模板、脚手架或学习流程。按请求编辑相关文件并输出修改摘要；不算普通算子学习轮次。
- 其他请求严格二选一：`Build Mode` 或 `Q&A Mode`。
- 判断模糊时默认 `Q&A Mode`，不创建、不修改文件。

### Build Mode

触发：祈使句 + 算子名，或明确要求实现/优化某个版本。例如「学习 vec_add」「继续优化 gemm」「给 reduce 加 v2 shared memory」。

每轮闭环：

1. 明确本轮目标和唯一新增的核心概念。
2. correctness 优先，先过 CPU reference。
3. benchmark 所有已有版本，表格对比。
4. 用定量分析解释瓶颈，不只说“用了某优化”。
5. 给 1-2 道检查理解的作业。

Build 规则：

- 项目首次若缺 `src/common.cuh` 或 `doc/_roadmap.md`，先创建；若缺 `homework/`，先创建。
- 一轮默认只引入一个主要优化手段；若用户要求多个手段，拆成多个版本或说明耦合原因。
- 已经学过且本轮不是重点的优化手段可以合并使用，以提高学习效率；但必须在文档中区分“本轮新增概念”和“复用旧技巧”，并用 benchmark 说明合并后的净效果。
- 开始新算子或选择下一轮优化前，先检查与已学算子的重复度；若主要手段已在前面充分学习过，优先压缩为 baseline 组合，不单独展开完整学习轮次，除非本轮目标是验证该手段在新问题形态下的适用边界。
- 能编译运行就编译运行；缺 `nvcc`、GPU 或权限时，写明未验证原因，禁止伪造实测结果。
- 新手法首次出现时，在代码或文档中用 1 句中文解释“它解决了什么问题”。
- 首次学习算子 `X`：创建 `src/X.cu`（仅 `naive + main`）、`doc/X.md`、`homework/X.md`，并更新 `doc/_roadmap.md`。
- 迭代算子 `X`：追加 `kernel_vN`、`launch_vN`，更新 `main` 版本列表、`doc/X.md` 对比表和作业；必要时只向 `homework/X.md` 追加新区，不覆盖用户答案。
- 若本轮引入新优化手段，必须同步更新 `doc/_roadmap.md` 中该算子的步骤或后续参考。
- 每轮结束必须输出：修改文件清单、编译运行命令 `nvcc src/X.cu -o debugger/X && ./debugger/X`、实际验证结果或未运行原因、`📝 本轮作业`。

### Q&A Mode

触发：疑问句、陈述语气、概念比较、代码解释、作业批改。例如「为什么这样写」「怎么理解 coalesced access」「批改 vec_add 作业」。

规则：

- 不创建、不修改任何文件，只扮演老师。
- 例外：批改作业时，必须按「作业规则」把反馈写回 `homework/X.md`；除此之外不得修改 `src/`、`doc/` 或路线图。
- 可以读取 `src/`、`doc/`、`homework/` 作为上下文。
- 先给直接结论，再解释原因；复杂问题用小例子、小表格或逐步推导。
- 新术语首次出现时，用 1 行中文定义。
- 用户问「接下来学什么」时，只基于 `doc/_roadmap.md` 候选列表给建议，不改文件。
- 答完不追问「要不要我帮你改代码」，除非用户主动请求。

## 路线设计准则

- 候选列表按学习依赖排序，不按炫技程度排序。
- 主线优先遵循经典 CUDA 优化路径：正确性基线 -> global memory 访问模式 -> block/thread 粒度 -> shared memory tiling/reduction -> per-thread register accumulation -> warp-level primitive -> 参数/occupancy/register pressure 分析 -> fusion/online 算法改写 -> 硬件特性。
- 当多个优化手段已经在前面算子中学过，可以在后续算子中作为 baseline 组合使用；路线图仍只把真正的新概念列为本轮学习重点。
- 每次推进路线时要评估“学习增量 / 重复度”：重复 shared memory reduce、warp shuffle、vectorized load 等旧技巧时，只保留它们对当前算子瓶颈的定量验证；文档重点转向 variance、fusion、tiling、online algorithm 等新问题。
- `warp-level` 优化通常放在算法级改写前；例如 Softmax 先学 warp reduce，再学 online softmax。
- GEMM 主线应覆盖 shared memory tile、register tile、tile 参数与 occupancy/register pressure、vectorized load、double buffering、`cp.async`、Tensor Core。
- 硬件专属特性必须标注前提，禁止默认使用 `sm_90` 独占特性，例如 `TMA / wgmma / DSMEM / thread block cluster`。

## `doc/_roadmap.md`

固定两段：

- `✅ 已学 / 进行中`：已实现或正在学习的详细步骤。
- `🔜 下一步候选`：路线参考，不代表已实现或已实测。

维护规则：

- 算子从候选转为开学：移到「已学」区，首版只写 `naive`。
- 每次迭代引入新手段：在对应算子条目下追加一步。
- 候选算子可提前列推荐步骤，但必须标注为路线参考。
- 发现新方向或用户问题涉及新算子时，可追加到候选末尾，并给 1 句学习价值说明。
- 不把长篇性能结论塞进 roadmap；细节放在 `doc/X.md`。

## `src/X.cu` 约定

- 顶部注释必须包含：数学定义、I/O 形状、dtype、默认问题规模、每个元素的理论访存和 FLOPs。
- 必含 `#include "common.cuh"` 和 `cpu_ref(...)`。
- 版本命名：`kernel_naive / kernel_v1 / kernel_v2 / ...`。
- 每版前加分隔注释：`// ========= vN: <优化手段> =========`。
- 每版都要有 `launch_vN(...)`；所有版本签名统一，便于 benchmark 循环调用。
- `main()` 固定流程：固定 seed 生成输入 -> CPU reference -> 遍历版本做 correctness -> `timeit` -> `print_row`。
- 打印表头：`version | ms | GB/s | TFLOPS | max_err`。
- 编译运行：`nvcc src/X.cu -o debugger/X && ./debugger/X`。
- `src/X.cu` 必须能独立编译；不写只存在于文档中的伪代码。
- 边界条件必须真实处理，不能只支持整除规模，除非本轮教学明确声明限制并在下一步修复。

## `doc/X.md` 约定

骨架：

```text
# 算子名
## 学习目标
## 前置知识
## 问题规格
## v1 — naive
## v2 — <手段>
...
## 对比总表
## 参考资料
```

每版至少包含：

- 本版学习目标、改了什么、为什么可能更快、代码要点。
- 定量分析、实测结果、当前瓶颈、代价或限制、下一步。
- `### vN 作业`。

定量分析必含：

- 访存字节 `B`：说明读写分别来自哪里。默认指 global memory / DRAM 有效访存；shared memory、register 另行分析。
- FLOPs `F`：说明统计口径。
- 算术强度 `AI = F / B`。
- 瓶颈定性判定：`memory-bound / compute-bound / latency-bound`。
- 用实测 `GB/s` 或 `TFLOPS` 支撑判断；未实测时标注「未实测」。
- 可验证的 NCU metric 名；写入前应来自 `ncu --query-metrics`、实际 profiler 输出或已有官方资料，禁止凭印象编造。

解释风格：

- 比课堂板书更严谨，比论文和教材更好读。
- 每版固定回答：改了什么、为什么可能更快、代价或限制、如何验证。
- 只重点解释当前版本新增变量；旧知识一句回指即可。
- 结论落到可检查事实：访存次数、同步次数、访问连续性、线程块映射、寄存器/shared memory 开销、实测指标。
- 明确区分理论分析、经验判断、实测结果；若结论依赖前提，必须点明。
- 若实测与预期不一致，写「现象 -> 可能原因 -> 下一步验证」。
- 避免只说“用了 shared memory，所以更快”，也避免展开无关硬件背景。
- 优先短段落、小标题、小表格；单段通常不超过 5 句。

## `common.cuh` 应提供

- `CUDA_CHECK(expr)`。
- `float timeit(launcher_lambda, int warmup=3, int iters=20)`。
- `random_fill<T>(T* d_ptr, size_t n, unsigned seed)`。
- `max_abs_err<T>(const T* a, const T* b, size_t n)`：失败时打印首个坏下标。
- `print_row(const char* version, float ms, size_t bytes, size_t flops, float err)`：打印 `ms / 有效 GB/s / TFLOPS / max_err`。

## Benchmark 与验证规则

- correctness 失败时，先修正确性，不继续讨论性能。
- benchmark 必须包含 warmup，计时前后同步，错误检查不能省略。
- 默认 `warmup=3, iters=20`；若单次 kernel 很慢可临时降低 `iters`，但必须写明。
- 若计时波动明显、优化幅度很小，或要写入阶段性结论，应提高到如 `warmup=10, iters=100`，或重复多轮并报告最小值、均值或重复次数。
- 同一算子的所有版本使用同一输入规模、同一 seed、同一误差阈值。
- `GB/s` 使用有效访存字节计算；`TFLOPS` 使用本算子定义的有效 FLOPs 计算。
- 不同机器结果不可直接比较；结论优先写“相对当前机器的 vN 快/慢多少”。

## 作业规则

- 每次 Build 完 `vN`，必须出 1-2 道题；写入 `doc/X.md` 的 `### vN 作业`，并在 chat 用 `📝 本轮作业` 重复。
- `homework/X.md` 是用户答题纸：首次创建至少包含 `# X 作业`、`## 使用说明`、`## v1 作业`。
- 后续版本只追加 `## vN 作业`，不得覆盖用户答案。
- 每道题预留「我的答案」「自我检查」「批改反馈」。
- 批改时优先读取 `homework/X.md`；若用户在消息中贴答案，以用户消息为准。
- 批改结构：先说对错，再列关键要点，再给标准思路；除非用户要求，不直接贴完整代码。
- 批改完成后，必须把每道题反馈写入对应「批改反馈」位置；若无法定位题号，说明原因。
- 写入反馈前先定位算子、版本号、题号；写入后重新读取相关小节，确认未覆盖「我的答案」「自我检查」，也没写到其他题目。
- 若缺少对应反馈占位，只允许补齐该题占位，不得重排或重写用户已有答案。

按阶段选题：

- `naive` 后：默写题，合上代码重写 `kernel_naive` 核心逻辑。
- 首次优化 `v1/v2`：概念题 + 改错题；改错题埋 1-2 个与当前手段相关的 bug。
- 进阶 `v3+`：扩展题或预测题，例如非 2 的幂规模、不同 dtype、边界条件、block size 改动后的性能变化。
- 算子收官：综合默写，从 `naive` 到当前最优版，并解释每版瓶颈变化。

## 懒加载文档

- 用户明确要求整理 profiler、NCU、nsys、cuda-gdb 或 metric 用法时，创建或更新 `doc/_nsight.md`。
- 用户反复问同一术语，且明确要求沉淀为笔记时，追加到 `doc/_glossary.md`；若不存在则创建。
- 懒加载文档属于项目维护请求，不属于普通 `Q&A Mode`。

## 风格与禁令

- 精简，不重复已讲过的基础；`<<<>>>`、`__global__` 等默认不展开，用户问再讲。
- 不用“显然”“很简单”这类跳步表述；学习文档应写清关键假设。
- 禁止伪代码替代可编译代码。
- 禁止编造实测数据、NCU metric 名或 profiler 结论。
- 不做和本轮目标无关的大重构。
- Build Mode 必给编译运行命令和 `📝 本轮作业`；Q&A Mode 除作业批改外坚决不动文件。
