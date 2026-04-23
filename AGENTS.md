# CUDA 算子渐进优化学习项目

## 项目结构
```text
root/
├── AGENTS.md
├── src/
│   ├── common.cuh
│   ├── vec_add.cu
│   ├── gemm.cu
│   └── ...
├── homework/
│   ├── vec_add.md
│   ├── gemm.md
│   └── ...
└── doc/
    ├── _roadmap.md
    ├── vec_add.md
    ├── gemm.md
    ├── _nsight.md
    └── _glossary.md
```

## 工作模式
每轮严格二选一；判断模糊时默认 `Q&A Mode`。

### Build Mode
触发：祈使句 + 算子名，如「学习 vec_add」「继续优化」「加 v2 用 shared memory」「重构 gemm v3」。

规则：
- 项目首次：若 `src/common.cuh` 或 `doc/_roadmap.md` 不存在，先创建再继续。
- 若 `homework/` 不存在，先创建；用于存放用户作业，不和 `doc/` 混用。
- 算子首次：
  - 创建 `src/X.cu`：仅 `naive + main`
  - 创建 `doc/X.md`：问题规格 + `v1` 小节 + `v1` 作业
  - 创建 `homework/X.md`：作为该算子的作业答题纸
  - 更新 `doc/_roadmap.md`：若 `X` 在「🔜 下一步候选」，移到「✅ 已学 / 进行中」并展开当前步骤；否则直接加到「已学」
- 迭代：
  - 在 `src/X.cu` 追加 `kernel_vN`、`launch_vN`，并更新 `main` 的版本列表
  - 在 `doc/X.md` 追加 `vN` 小节、更新对比总表、追加 `vN` 作业
  - 必要时更新 `homework/X.md` 模板，追加新的作业答题区，但保留用户已有内容
  - 若本版引入新优化手段，在 `doc/_roadmap.md` 的该算子条目下追加一步
- 每轮结束必须输出：修改文件清单、编译运行命令、`📝 本轮作业`

### Q&A Mode
触发：疑问句或陈述语气，如「为什么…」「怎么理解…」「X 和 Y 区别」「这行为啥这么写」「我作业写成这样对吗」。

规则：
- 不创建、不修改任何文件，只扮演老师。
- 由浅入深；必要时用小例子或小表格；新术语首次出现时用 1 行中文解释。
- 批改作业：优先读取 `homework/X.md` 中用户填写的内容；先说对错，再说要点，再给标准思路；除非用户要求，否则不直接贴完整代码。
- 当用户说「批改 vec_add 作业」「我写完 gemm 作业了」这类话时，默认读取对应的 `homework/X.md` 进行批改；若用户在消息里直接贴答案，则以用户消息为准。
- 答完不追问「要不要我帮你改代码」，除非用户主动请求。

## `doc/_roadmap.md`
固定两段：
- `✅ 已学 / 进行中`：详细步骤
- `🔜 下一步候选`：仅简介

维护规则：
- 算子从候选转为开学：移到「已学」区，首版只写 `naive` 一步
- 每次迭代引入新手段：在对应算子条目下追加一步
- 「下一步候选」初始由 Claude 预填常见算子；发现新方向或用户问题涉及新算子时，追加到候选列表末尾
- 用户问「接下来学什么」时，仅在 `Q&A Mode` 基于候选列表给建议，不改文件

## `src/X.cu` 约定
- 顶部注释：数学定义、I/O 形状、dtype、默认问题规模
- 必含：`#include "common.cuh"`、`cpu_ref(...)`
- 版本命名：`kernel_naive / kernel_v1 / kernel_v2 / ...`
- 每版前加分隔注释：`// ========= vN: <优化手段> =========`
- 每版都要有 `launch_vN(...)`；所有版本签名统一，便于 benchmark 循环调用
- `main()` 固定流程：
  1. 固定 seed，用 `random_fill` 生成输入
  2. 用 `cpu_ref` 生成 ground truth
  3. 遍历所有版本：正确性检查 → `timeit` → `print_row`
  4. 打印表头：`version | ms | GB/s | TFLOPS | max_err`
- 编译运行：`nvcc src/X.cu -o debugger/X && ./debugger/X`
- 新手法首次出现时，加 1 行中文注释解释“为什么这么做”

## `doc/X.md` 约定
骨架：
```text
# 算子名
## 问题规格
## v1 — naive
## v2 — <手段>
...
## 对比总表
## 作业
## 参考资料
```

每版至少包含：
- 改动
- 代码要点
- 定量分析
- 实测结果
- 瓶颈
- 下一步

定量分析必含：
- 访存字节 `B`
- FLOPs `F`
- 算术强度 `AI = F / B`
- 瓶颈定性判定：`memory-bound / compute-bound / latency-bound`
- 用实测 `GB/s` 或 `TFLOPS` 支撑判断
- 可验证的 NCU metric 名；禁止编造

解释风格：
- 目标：比课堂板书更严谨，比论文和教材更好读
- 每版固定回答：改了什么、为什么可能更快、代价或限制、如何验证
- 只重点解释当前版本新增的变量；旧知识只做一句回指
- 结论尽量落到可检查事实：访存次数、同步次数、访问连续性、线程块映射、寄存器或 shared memory 开销、实测指标
- 明确区分理论分析、经验判断、实测结果
- 数学推导只保留支撑当前分析所需的最小集合；既不省略关键量，也不做冗长推导
- 优化手段首次出现时，先用 1 句直白中文定义
- 描述瓶颈时必须说清“主要受什么限制、为什么”
- 若结论依赖前提，必须点明前提
- 若实测与预期不一致，必须写「现象 → 可能原因 → 下一步验证」
- 避免两种失衡：
  - 只说“用了 shared memory，所以更快”
  - 展开过多硬件背景或公式推导，打断主线
- 优先短段落、小标题、小表格；单段通常不超过 5 句

## `common.cuh` 应提供
- `CUDA_CHECK(expr)`
- `float timeit(launcher_lambda, int warmup=10, int iters=100)`
- `random_fill<T>(T* d_ptr, size_t n, unsigned seed)`
- `max_abs_err<T>(const T* a, const T* b, size_t n)`：失败时打印首个坏下标
- `print_row(const char* version, float ms, size_t bytes, size_t flops, float err)`：打印 `ms / 有效 GB·s⁻¹ / TFLOPS / max_err`

## 作业规则
每次 Build 完 `vN`，必须出 1–2 道题；写入 `doc/X.md` 的 `### vN 作业`，并在 chat 用 `📝 本轮作业` 重复。

`homework/X.md` 规则：
- 文件名与算子名一致，如 `homework/vec_add.md`
- 用途是让用户写代码块、回答概念题、记录自己的分析过程
- 首次创建时提供基础骨架，至少包含：`# X 作业`、`## 使用说明`、`## v1 作业`，后续版本按需追加 `## vN 作业`
- 若用户已填写内容，后续迭代只能追加新区块或补充题目占位，不得覆盖用户答案
- 用户请求批改作业时，默认把该文件视为主要批改对象

按阶段选题：
- `naive` 后：默写题，合上代码重写 `kernel_naive` 核心循环
- 首次优化 `v1/v2`：概念题 + 改错题；改错题应埋 1–2 个与当前手段相关的 bug
- 进阶 `v3+`：扩展题（非 2 的幂规模、不同 dtype、边界条件）或预测题（改参数后性能如何变化、为什么）
- 算子收官：综合默写，从 `naive` 一路写到当前最优版

## 懒加载文档
- 用户首次问 profile、NCU、nsys、cuda-gdb、怎么看 metric：创建 `doc/_nsight.md`，内容含常用命令、关键 metric 清单、`cuda-gdb` 基本调试
- 用户反复问同一术语：追加到 `doc/_glossary.md`；若不存在则创建

## 风格与禁令
- 中文讲解；CUDA 标识符和指令保留英文
- 精简，不重复已讲过的基础；`<<<>>>`、`__global__` 等默认不展开，用户问再讲
- 禁止：伪代码、编造 NCU metric 名、使用 `sm_90` 独占特性（`TMA / wgmma / DSMEM / thread block cluster`）
- Build Mode 必给编译运行命令和 `📝 本轮作业`；Q&A Mode 坚决不动文件
