# 中文测试报告

## 总览

- 测试状态：全部通过
- 测试总数：17
- 通过：17
- 失败：0
- 跳过：0
- 总耗时：0.54 秒

## 模块覆盖说明

| 模块 | 测试目的 | 通过 | 失败 | 跳过 | 耗时 (秒) |
|---|---|---:|---:|---:|---:|
| **有限体积离散** (`tests/test_discretization.py`) | 检查 harmonic face diffusion、测试专用 Dirichlet 有限体积算子是否对有真解问题收敛，并检查 Wilson limiter 是否降低原始扩散系数。 | 3 | 0 | 0 | 0.047 |
| **驱动器与实验统计** (`tests/test_integration_experiments.py`) | 检查迭代平均值、标度指数辅助函数，以及一个极短时间积分是否正确记录线性/非线性迭代历史。 | 2 | 0 | 0 | 0.005 |
| **线性求解器与多重网格** (`tests/test_linear_multigrid.py`) | 检查 GMRES/CG 是否能解小型 SPD 系统；检查 Picard 线性算子的线性性；检查一次 V-cycle 是否降低残差；检查 MG 作为预条件器是否减少 GMRES 迭代。 | 4 | 0 | 0 | 0.038 |
| **Picard/JFNK/NK2 与时间步控制** (`tests/test_methods_driver.py`) | 检查 midpoint 状态构造、Jacobian-free matvec、Picard 线性化、一个小网格 NK2 时间步收敛，以及 eta/dt 更新规则。 | 5 | 0 | 0 | 0.006 |
| **问题设置与物理模型** (`tests/test_problem_physics.py`) | 检查网格、材料区域、边界条件、M1/M2/M3 的储能项、质量系数、扩散系数和 Wilson limiter。 | 3 | 0 | 0 | 0.006 |

## 逐项测试说明

| 测试项 | 所属模块 | 检查内容 | 结果 | 耗时 (秒) |
|---|---|---|---|---:|
| `test_dirichlet_finite_volume_manufactured_solution_converges` | 有限体积离散 | 有限体积离散在有解析解的 Dirichlet 问题上是否随网格加密收敛。 | 通过 | 0.001 |
| `test_harmonic_face_diffusion_for_jump_coefficients` | 有限体积离散 | 材料跳跃处 face diffusion 是否使用 harmonic mean。 | 通过 | 0.046 |
| `test_wilson_limiter_on_faces_reduces_raw_coefficients` | 有限体积离散 | M3 face-based limiter 是否不会放大原始扩散系数。 | 通过 | 0.000 |
| `test_iteration_summary_helpers` | 驱动器与实验统计 | 实验表格中的平均迭代数和标度指数统计函数是否工作正常。 | 通过 | 0.001 |
| `test_short_driver_run_records_histories` | 驱动器与实验统计 | 短时间积分是否正确记录时间、dt、线性迭代、非线性迭代和残差历史。 | 通过 | 0.005 |
| `test_gmres_and_cg_solve_small_spd_system` | 线性求解器与多重网格 | GMRES 和 CG 对小型线性系统是否给出正确解。 | 通过 | 0.001 |
| `test_multigrid_preconditioning_reduces_gmres_iterations` | 线性求解器与多重网格 | MG 预条件器是否能减少 GMRES 迭代数。 | 通过 | 0.027 |
| `test_multigrid_vcycle_reduces_picard_residual` | 线性求解器与多重网格 | 一次多重网格 V-cycle 是否能降低 Picard 线性系统残差。 | 通过 | 0.002 |
| `test_picard_operator_is_linear` | 线性求解器与多重网格 | 冻结系数后的 Picard operator 是否满足线性性。 | 通过 | 0.008 |
| `test_eta_and_dt_update_rules` | Picard/JFNK/NK2 与时间步控制 | eta 计算和时间步增长限制是否符合设计。 | 通过 | 0.000 |
| `test_jacobian_free_matvec_matches_quadratic_directional_derivative` | Picard/JFNK/NK2 与时间步控制 | JFNK 有限差分 matvec 是否能近似已知导数。 | 通过 | 0.000 |
| `test_midpoint_states_follow_temperature_midpoint_rule` | Picard/JFNK/NK2 与时间步控制 | NK2 的 implicit midpoint 状态是否按论文温度中点规则构造。 | 通过 | 0.000 |
| `test_nk2_residual_and_step_converge_on_tiny_grid` | Picard/JFNK/NK2 与时间步控制 | 小网格上单个 NK2 时间步是否能把非线性残差降到容差以下。 | 通过 | 0.004 |
| `test_picard_linearization_is_linear_and_shape_preserving` | Picard/JFNK/NK2 与时间步控制 | Picard 线性化接口是否线性，且预条件器返回维度正确。 | 通过 | 0.001 |
| `test_diffusion_models_and_wilson_limiter_are_monotone` | 问题设置与物理模型 | 扩散系数随温度的幂次是否正确，Wilson limiter 是否确实限制扩散系数。 | 通过 | 0.000 |
| `test_grid_material_and_boundary_setup` | 问题设置与物理模型 | 网格、材料分区、初值和边界类型是否符合论文二维问题设置。 | 通过 | 0.005 |
| `test_physics_storage_and_mass_models` | 问题设置与物理模型 | M1/M2/M3 的储能量 Q(E) 和 Q'(E) 是否与模型定义一致。 | 通过 | 0.001 |

## 结论

当前测试说明：基础物理模型、有限体积离散、线性求解器、多重网格预条件、Picard 线性化、JFNK/NK2 时间步和实验统计辅助函数都通过了独立验证。
这些测试不能保证论文表格迭代数完全一致，但可以帮助定位代码错误：如果后续修改某个技术模块导致测试失败，就能更快知道问题出在哪一层。
