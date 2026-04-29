# 二维非线性辐射扩散方程复现实验代码

本项目用于复现论文中的二维多材料非线性辐射扩散实验，重点关注 Picard 线性化方法与 Newton-Krylov/NK2 方法在不同模型、网格和时间步控制参数下的迭代表现。当前代码已经完成基本的技术拆分：物理模型、有限体积离散、线性求解器、多重网格预条件器、非线性时间步方法、实验驱动、checkpoint、测试和报告生成分别放在不同模块中，便于逐层排查误差来源。

## 当前状态

当前代码已经具备完整运行链路：

```text
建模
  -> 物理系数
  -> 有限体积离散
  -> Picard / NK2 时间步
  -> GMRES / CG 线性求解
  -> Picard 型多重网格预条件器
  -> eta 时间步控制
  -> checkpoint 保存/恢复
  -> 实验统计与论文对比报告
```

目前已经生成了 NK2 checkpoint 结果，并可自动和论文 Tables XIII-XVI 对比。对比报告位于：

- `output/paper_comparison.html`
- `output/paper_comparison.csv`

报告采用三线表风格，包含论文值、本代码值、差值、相对误差、完成状态和 Scaling exponent。Scaling exponent 按论文定义计算：计算工作量满足 `work ~ N^s`，由于一次迭代本身需要 `O(N)` 工作量，所以代码中等价为先拟合平均迭代数关于自由度 `N = nx * ny` 的幂次，再加 1。

## 代码结构

| 文件/目录 | 作用 |
|---|---|
| `config.py` | 定义运行参数 `RunConfig` 和求解器参数 `SolverConfig`。 |
| `problem.py` | 构造二维网格、材料分区、边界条件和初始能量场。 |
| `physics.py` | 实现 M1/M2/M3 储能项、质量系数、扩散系数、材料因子和 Wilson limiter。 |
| `discretization.py` | 实现有限体积离散、face diffusion、边界通量、冻结扩散算子和测试用 Dirichlet 算子。 |
| `solvers.py` | 实现 GMRES、CG、Jacobi/GS/SGS smoother、多重网格 V-cycle 和 `PicardMGPreconditioner`。 |
| `methods.py` | 实现 Picard step、NK2 step、JFNK matvec、midpoint 状态构造和阻尼更新。 |
| `driver.py` | 实现时间推进、eta 控制、迭代历史记录、checkpoint 保存/恢复。 |
| `test.py` | 当前主要实验入口，可运行 single 或 batch，并可将结果追加到 `output/test_results.csv`。 |
| `experiments.py` | 论文表格风格的批量实验入口和平均迭代/标度指数辅助函数。 |
| `compare_nk2_experiments.py` | 诊断脚本，用于比较不同 NK2/MG/Picard 参数变体。 |
| `compare_paper_results.py` | 从 checkpoint 读取 NK2 结果，并生成论文三线表对比报告。 |
| `generate_test_report.py` | 运行 pytest 并生成中文 Markdown/HTML 测试报告。 |
| `tests/` | 单元测试目录，按技术模块验证关键代码。 |
| `checkpoints/` | 保存实验中间状态，支持恢复长时间实验。 |
| `output/` | 保存测试报告、论文对比报告和实验输出文件。 |

## 核心模块说明

### 1. 问题设置与物理模型

`problem.py` 负责构造论文中的二维多材料几何：

- 背景材料 `Z = 10`。
- 左下矩形区域 `Z = 20`。
- 右下矩形区域 `Z = 100`。
- 右上圆形区域 `Z = 50`。
- 初值为 `E = 1`。
- 左右边界为 Milne/Robin 类型边界，上下边界为对称边界。

`physics.py` 实现 M1/M2/M3 三类模型，包括能量到温度的转换、储能项、冻结质量系数、原始扩散系数和 Wilson limiter。

### 2. 有限体积离散

`discretization.py` 将连续扩散项离散为 cell-centered 有限体积算子。主要内容包括：

- harmonic mean face diffusion；
- Milne/Robin 与对称边界通量；
- Wilson limiter 的 face-based 处理；
- Picard/MG 使用的冻结扩散算子；
- 测试用 Dirichlet manufactured solution 算子。

### 3. 线性求解器与多重网格

`solvers.py` 中包含：

- `gmres(...)` 和 `cg(...)`；
- Jacobi、Gauss-Seidel、symmetric Gauss-Seidel smoother；
- full-weighting restriction 和 piecewise-constant prolongation；
- Picard 型多重网格 V-cycle；
- `PicardMGPreconditioner`，作为 NK2/JFNK 中 GMRES 的预条件器。

当前 MG 的目标是贴近论文中的 Picard-type multigrid preconditioning，而不是通过人为增强预条件器强行匹配论文迭代数。

### 4. Picard 与 NK2

`methods.py` 是非线性方法核心：

- `picard_step(...)`：冻结系数 Picard 迭代。
- `nk2_step(...)`：Newton-Krylov/NK2 时间步。
- `residual_nk2(...)`：NK2 非线性残差。
- `jacobian_free_matvec(...)`：Jacobian-free 有限差分矩阵向量乘。
- `build_midpoint_states(...)`：implicit midpoint 状态构造。
- `_damped_update(...)`：阻尼更新，目前支持不同范数设置。

`driver.py` 在时间循环中调用这些单步方法，并用 `eta` 控制时间步增长。

## 如何运行实验

### 运行主实验

```powershell
python test.py
```

常用参数在 `test.py` 顶部修改：

- `MODE = "single"` 或 `"batch"`。
- `METHOD = "nk2"` 或 `"picard"`。
- `MODEL = "M1"`、`"M2"`、`"M3"`。
- `ETA = 0.10` 或 `0.50`。
- `NX`、`NY` 控制网格规模。
- `RESUME_FROM_CHECKPOINT` 控制是否从 checkpoint 恢复。
- `CHECKPOINT_INTERVAL` 控制保存间隔。

批量实验由以下列表控制：

```python
BATCH_METHODS = ["nk2", "picard"]
BATCH_MODELS = ["M1", "M2", "M3"]
BATCH_ETAS = [0.10, 0.50]
BATCH_GRIDS = [32, 64, 128]
```

当前 checkpoint 文件名已经区分方法，例如：

```text
checkpoints/M1_eta0.1_grid64_methodnk2.npz
checkpoints/M1_eta0.1_grid64_methodpicard.npz
```

因此 NK2 和 Picard 不会再互相覆盖。

### 保存实验结果

`test.py` 每完成一个算例，会将摘要结果追加写入：

```text
output/test_results.csv
```

该文件适合用 Excel 打开，也适合之后和论文结果做自动对比。如果该文件被删除，仍可以通过 checkpoint 恢复主要迭代统计。

### 从 checkpoint 恢复

如果要恢复某个算例，需要让 `test.py` 顶部参数与 checkpoint 文件名一致。例如恢复：

```text
checkpoints/M2_eta0.5_grid128_methodnk2.npz
```

应设置：

```python
MODE = "single"
METHOD = "nk2"
MODEL = "M2"
ETA = 0.50
NX = 128
NY = 128
RESUME_FROM_CHECKPOINT = True
```

运行后若成功，会看到类似：

```text
[resume] loaded checkpoint from ...
```

## 论文结果对比

生成论文三线表风格对比报告：

```powershell
python compare_paper_results.py
```

输出：

- `output/paper_comparison.html`
- `output/paper_comparison.csv`

该脚本当前对比论文 Tables XIII-XVI，即 Newton-Krylov 方法在 `eta = 0.10` 和 `eta = 0.50` 下的平均线性/非线性迭代数。脚本会从：

```text
checkpoints/*_methodnk2.npz
```

读取本代码 NK2 结果，并自动计算：

- 论文线性迭代；
- 本代码线性迭代；
- 线性差值和相对误差；
- 论文非线性迭代；
- 本代码非线性迭代；
- 非线性差值和相对误差；
- 是否完整跑到论文终止时间；
- Scaling exponent。

注意：如果某个 checkpoint 没有跑到论文对应终止时间，报告中会标记为“未完成/缺失”，并且不参与本代码 Scaling exponent 拟合。

## 测试

运行全部测试：

```powershell
python -m pytest -q
```

当前测试按技术模块拆分：

| 测试文件 | 验证内容 |
|---|---|
| `tests/test_problem_physics.py` | 网格、材料、边界、M1/M2/M3 储能项、质量系数、扩散系数和 Wilson limiter。 |
| `tests/test_discretization.py` | harmonic face diffusion、有限体积收敛性、M3 limiter 是否限制扩散系数。 |
| `tests/test_linear_multigrid.py` | GMRES/CG、Picard 线性算子、V-cycle 降残差、MG 预条件是否减少 GMRES 迭代。 |
| `tests/test_methods_driver.py` | midpoint 状态、JFNK matvec、Picard 线性化、NK2 小网格时间步、eta/dt 更新。 |
| `tests/test_integration_experiments.py` | 驱动器历史记录、平均迭代统计和标度指数辅助函数。 |

生成中文测试报告：

```powershell
python generate_test_report.py
```

输出：

- `output/test_report.md`
- `output/test_report.html`

HTML 报告是自包含文件，不依赖外部 CDN，也不需要安装 `pytest-html`。如果测试失败，脚本仍会生成报告并返回 pytest 的退出码。

## 当前复现结论

从当前 NK2 对比结果看，部分算例已经比较接近论文：

- M2 的线性/非线性迭代数整体较接近论文；
- M1 的非线性迭代数较接近，但线性迭代数偏高；
- M3 在部分网格下非线性迭代数接近，但线性迭代数随网格加密偏高；
- 少数较大算例 checkpoint 尚未完整跑到论文终止时间，需要继续运行或重新生成正式结果。

这些偏差说明当前实现已经能跑通论文主要流程，但仍不能认为与论文实现完全等价。可能敏感来源包括：

- Picard 线性化强弱；
- M3 下 Wilson limiter 与 MG 预条件器的一致性；
- JFNK 有限差分扰动尺度；
- 阻尼范数和阻尼策略；
- 多重网格 coarse operator、smoother 参数和边界处理；
- 论文中没有完全展开的实现细节，例如具体范数、预条件器内部迭代细节和边界离散方式。

## 文件整理建议

推荐保留：

- 核心 `.py` 文件；
- `tests/`；
- `README.md`；
- `paper.txt`；
- `checkpoints/`，如果还需要恢复实验；
- `output/paper_comparison.html` 和 `output/paper_comparison.csv`；
- `output/test_report.html` 和 `output/test_report.md`。

## 依赖环境

主要依赖：

- Python 3
- NumPy
- pytest
- SciPy，只有使用 `gmres_scipy_right(...)` 相关后端时需要

通常运行当前默认测试和主流程时，确保 NumPy 与 pytest 可用即可。
