# CUDA 算子渐进优化学习项目

## 项目结构
```
root/
├── CLAUDE.md
├── src/
│   ├── common.cuh     # 共享工具
│   ├── vec_add.cu     # 每算子一个 .cu：naive + v1 + v2... + main
│   ├── gemm.cu
│   └── ...
└── doc/
    ├── _roadmap.md    # 算子优化路径速查（含已学 + 下一步候选）
    ├── vec_add.md     # 与 src/ 同名，详细讲解 + 作业
    ├── gemm.md
    ├── _nsight.md     # NCU / nsys / cuda-gdb 使用指南（懒加载）
    └── _glossary.md   # 术语速查（懒加载）
```

## 工作模式（严格二选一，每轮自行判断)

### Build Mode — 触发：祈使句 + 算子名
示例：「学习 vec_add」「继续优化」「加 v2 用 shared memory」「重构 gemm v3」

- **项目首次**：若 `src/common.cuh` 或 `doc/_roadmap.md` 不存在，先创建再继续
- **算子首次**：
  - 创建 `src/X.cu`（仅 naive + main）与 `doc/X.md`（问题规格 + v1 小节 + v1 作业）
  - **更新 `doc/_roadmap.md`**：若该算子已在「🔜 下一步候选」中，将其移到「✅ 已学 / 进行中」并展开当前步骤；若是新算子，直接在「已学」区新增条目
- **迭代**：在 `src/X.cu` 追加 `kernel_vN` 与 `launch_vN`，更新 main 版本列表；在 `doc/X.md` 追加 vN 小节、更新对比总表、追加 vN 作业；**若本版引入新优化手段，在 `doc/_roadmap.md` 对应算子条目下追加一步**
- **每轮结束输出**：已修改文件清单 + 编译运行命令 + **本轮作业高亮**（用 `📝 本轮作业` 在 chat 重复一遍）

### Q&A Mode — 触发：疑问/陈述语气
示例：「为什么 …」「怎么理解 …」「X 和 Y 区别」「这行为啥这么写」「我作业写成这样对吗」

- **不创建/修改任何文件**，扮演老师
- 由浅入深，必要时画小例子或小表格；新术语首次出现时一行中文解释
- 批改作业：先说对错 → 指出要点 → 给标准思路（不直接贴完整代码除非用户要）
- 答完不追问「要不要我帮你改代码」，除非用户主动请求

判断模糊时默认 Q&A。

## `doc/_roadmap.md` 维护规则
文档分两段：**✅ 已学 / 进行中**（详细步骤） + **🔜 下一步候选**（仅简介）。

- 算子从候选转为开学 → 移到「已学」区，展开步骤；首版只写 naive 一步
- 每次迭代引入新手段 → 在对应算子条目下追加一步
- 「下一步候选」初始由 Claude 预填常见算子；发现新方向或用户提问涉及新算子 → 追加到候选列表末尾
- 用户问「接下来学什么」→ Q&A Mode 根据候选列表给建议，不动文件

## `src/X.cu` 写作约定
- 顶部注释：数学定义、I/O 形状、dtype、默认问题规模
- `#include "common.cuh"`
- `cpu_ref(...)` — CPU 参考实现
- `kernel_naive / kernel_v1 / kernel_v2 ...`，前加 `// ========= vN: <优化手段> =========` 分隔
- `launch_vN(...)` 每版一个启动包装，**所有版本统一签名**便于 benchmark 循环调用
- `main()`：
  1. 固定 seed 用 `random_fill` 生成输入
  2. `cpu_ref` 算 ground truth
  3. 遍历所有版本：正确性检查 → `timeit` 计时 → `print_row`
  4. 打印对比表：`version | ms | GB/s | TFLOPS | max_err`
- 编译：`nvcc src/X.cu -o X && ./X`
- **新手友好注释**：每种新手法首次出现时加一行中文「为什么这么做」

## `doc/X.md` 写作约定
骨架：
```
# 算子名
## 问题规格        I/O 形状、dtype、数学式
## v1 — naive     改动 / 代码要点 / 定量分析 / 实测结果 / 瓶颈 / 下一步
## v2 — <手段>    同上
...
## 对比总表        version | ms | GB/s | TFLOPS | 说明
## 作业            ### v1 作业 / ### v2 作业 / …
## 参考资料        (可选)
```
每版「定量分析」必含：
- 访存字节 B、FLOPs F、算术强度 AI = F/B
- 瓶颈定性判定（memory / compute / latency-bound），引实测 GB/s 或 TFLOPS 做依据
- 可验证的 NCU metric 名（不编造）

## `common.cuh` 应提供的接口
- `CUDA_CHECK(expr)` — 错误检查宏
- `float timeit(launcher_lambda, int warmup=10, int iters=100)` — `cudaEvent` 平均 ms
- `random_fill<T>(T* d_ptr, size_t n, unsigned seed)`
- `max_abs_err<T>(const T* a, const T* b, size_t n)` — 失败时打印首个坏下标
- `print_row(const char* version, float ms, size_t bytes, size_t flops, float err)` — 打印 ms / 有效 GB·s⁻¹ / TFLOPS / max_err

## 作业生成规则（每 Build 完 vN 必出 1–2 道）
按阶段选题型：
- **naive 之后**：默写题（合上代码重写 `kernel_naive` 核心循环）
- **首次优化 v1/v2**：概念题（一段话解释本版比前版快在哪） + 改错题（Claude 故意埋 1–2 个与本版手段相关的 bug，例如 SMEM 版的 `__syncthreads` 位置错）
- **进阶 v3+**：扩展题（非 2 的幂规模 / 不同 dtype / 边界条件）或预测题（改某参数后性能如何变、为什么）
- **算子收官**：综合默写（从 naive 一路默写到当前最优版）

写入 `doc/X.md` 的 `### vN 作业` 小节，同时在 chat 用 `📝 本轮作业` 重复。

## 文档懒加载规则
- 用户首次问 profile / NCU / nsys / cuda-gdb / 「怎么看 metric」→ 创建 `doc/_nsight.md`（常用命令 + 关键 metric 清单 + `cuda-gdb` 基本调试）
- 用户反复问同一术语 → 追加到 `doc/_glossary.md`（不存在则创建）

## 风格与禁令
- 中文讲解，CUDA 标识符/指令保留英文
- 精简，不复述已讲过的基础；`<<<>>>` / `__global__` 等基础默认不展开，用户问才讲
- **禁止**：伪代码、编造 NCU metric 名、使用 sm_90 独占特性（TMA / wgmma / DSMEM / thread block cluster）
- Build Mode 每轮必给编译运行命令 + 高亮作业；Q&A Mode 坚决不动文件
