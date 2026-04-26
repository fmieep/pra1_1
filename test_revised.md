---
documentclass: ctexart
---

# A Multigrid Newton--Krylov Method for Multimaterial Equilibrium Radiation Diffusion

- 作者：William J. Rider, Dana A. Knoll, Gordon L. Olson
- 期刊：Journal of Computational Physics, 1999
- 关键词：equilibrium radiation diffusion, multimaterial, nonlinear diffusion, multigrid, Newton--Krylov, GMRES

---

## 1. 论文一句话总结

本文研究的是**多材料平衡辐射扩散方程**的全隐式、非线性收敛求解。作者把问题写成关于辐射能量密度 $E$ 的强非线性扩散方程，然后用 **Jacobian-free Newton--Krylov** 方法求解每个时间步内的非线性方程，其中 Krylov 线性子问题采用 **GMRES**，预条件器来自 **Picard 线性化**，而 Picard 线性系统的近似逆由一个 **multigrid V-cycle** 给出。

这篇文章的核心思想不是单独提出一个多重网格线性求解器，而是把

$$
\text{非线性辐射扩散} + \text{Picard 预条件} + \text{multigrid V-cycle} + \text{GMRES} + \text{Jacobian-free Newton}
$$

组合成一个可以处理强非线性、多材料间断系数和 flux limiter 的整体求解框架。

---

## 2. 具体方程问题：已知、未知与参数

### 2.1 从非平衡辐射扩散到平衡辐射扩散

论文先从非平衡辐射扩散方程出发：

$$
\frac{\partial E}{\partial t}
= \nabla \cdot \left(\frac{c}{3\kappa}\nabla E\right)
+ c\kappa(aT^4-E),
$$

$$
\frac{\partial C_vT}{\partial t}
= c\kappa(E-aT^4).
$$

其中：

- $E$：辐射能量密度；
- $T$：材料温度；
- $\kappa$：opacity，吸收不透明度；
- $a$：辐射常数；
- $c$：光速；
- $C_v$：材料比热相关系数。

本文进一步采用**平衡假设**

$$
E=aT^4,
$$

也就是辐射能量和材料温度在局部满足热平衡关系。于是 $T$ 不再是独立未知量，而可以由 $E$ 得到：

$$
T=\left(\frac{E}{a}\right)^{1/4}.
$$

论文为了简化量纲，取

$$
C_v=c=a=1,
$$

因此有

$$
T=E^{1/4}.
$$

把辐射能和材料能相加之后，得到论文真正求解的单变量平衡辐射扩散方程：

$$
\frac{\partial}{\partial t}\left[\left(\alpha+(1-\alpha)C_vE^{-3/4}\right)E\right]
=\nabla\cdot\left(D(E)\nabla E\right).
$$

为了书写方便，可以定义存储量

$$
Q(E)=\left(\alpha+(1-\alpha)C_vE^{-3/4}\right)E.
$$

在 $C_v=1$ 时，

$$
Q(E)=\alpha E+(1-\alpha)E^{1/4}.
$$

于是方程可写成

$$
\frac{\partial Q(E)}{\partial t}=\nabla\cdot(D(E)\nabla E).
$$

这就是论文的核心 PDE。

### 2.2 未知量是什么

在平衡辐射扩散模型中，真正需要数值求解的未知量是

$$
E(x,y,t).
$$

在离散网格上，就是每个网格单元中心的

$$
E_{i,j}^{n+1}.
$$

温度不是独立求解变量，而是通过

$$
T_{i,j}^{n+1}=\left(E_{i,j}^{n+1}\right)^{1/4}
$$

后处理得到。

所以从程序实现角度看，每个时间步要求解的是一个非线性代数方程组：

$$
F(E^{n+1})=0.
$$

其中 $E^{n+1}$ 是包含所有网格点未知量的大向量。

### 2.3 已知量是什么

一个完整算例中需要给定：

1. **计算区域**

   $$
   \Omega=[0,1]\times[0,1].
   $$

2. **初值**

   论文二维多材料算例采用

   $$
   E(x,y,0)=1.
   $$

3. **材料分布 $Z(x,y)$**

   不同材料通过原子序数 $Z$ 区分，进而影响扩散系数。论文的二维算例中：

   - $x\le 0.5,\ y\le 0.5$ 的矩形区域：$Z=20$；
   - $x\ge 0.75,\ y\le 0.25$ 的矩形区域：$Z=100$；
   - 圆形区域

     $$
     \sqrt{(x-0.75)^2+(y-0.75)^2}\le 0.15
     $$

     中：$Z=50$；
   - 其他区域：$Z=10$。

4. **边界条件**

   左右边界使用 Milne / Robin 型混合边界条件：

   $$
   F_{inc}=\frac12 D(T)\nabla E+\frac14E.
   $$

   二维算例中：

   - 左边界：$F_{inc}=2.5\times 10^3$；
   - 右边界：$F_{inc}=0.25$；
   - 上下边界：对称边界。

   这些通量对应的渐近边界能量大致为左端 $10^4$、右端 $1$。

5. **模型参数**

   包括 $\alpha$、$C_v$、扩散系数形式、是否使用 flux limiter、时间步控制参数 $\eta$、网格大小等。

### 2.4 参数有哪些

本文主要涉及下面几类参数。

| 参数 | 含义 | 在论文中的作用 |
|---|---|---|
| $E$ | 辐射能量密度 | 主要未知量 |
| $T$ | 材料温度 | 由 $T=E^{1/4}$ 得到 |
| $\alpha$ | 能量变量切换参数 | $\alpha=1$ 偏辐射能形式，$\alpha=0$ 偏材料能形式 |
| $C_v$ | 材料比热相关参数 | 论文简化取 $C_v=1$ |
| $Z$ | 材料原子序数 | 控制多材料扩散系数跳跃 |
| $\kappa$ | opacity | $D=c/(3\kappa)$ |
| $D$ | 扩散系数 | 非线性来源之一 |
| $D_L$ | flux-limited diffusion coefficient | 用于限制辐射传播速度 |
| $\eta$ | 单步相对能量变化控制量 | 动态调节时间步长 |
| $E_{floor}$ | 防止分母过小的能量底值 | 论文算例取 $E_{floor}=1$ |
| $\Delta t$ | 时间步长 | 由 $\eta$ 控制动态变化 |
| $h,\Delta x,\Delta y$ | 网格尺度 | 影响离散和线性系统难度 |

时间步控制采用

$$
\eta=\max_{i,j}\frac{|E_{i,j}^{n+1}-E_{i,j}^{n}|}{E_{i,j}^{n+1}+E_{floor}}.
$$

论文主要测试

$$
\eta=0.10\quad \text{和}\quad \eta=0.50.
$$

同时限制时间步每次增长不超过 $10\%$。

### 2.5 扩散系数的非线性来源

扩散系数写作

$$
D=\frac{c}{3\kappa}.
$$

由于 opacity 依赖温度，常见关系为

$$
\kappa\propto \frac{1}{T^3},
$$

所以

$$
D\propto T^3.
$$

又因为平衡条件下 $T=E^{1/4}$，所以

$$
D\propto E^{3/4}.
$$

多材料问题中还引入材料依赖项，论文中取

$$
D\propto Z^{-3}.
$$

如果使用 Wilson flux limiter，则扩散系数进一步变成

$$
D_L=\frac{1}{\frac{1}{D(T)}+\frac{|\nabla E|}{E}}.
$$

它的作用是：当梯度较小时回到普通扩散形式；当梯度很陡时，限制通量，避免传播速度不合理地超过物理上限。

---

## 3. 论文具体求解了哪些算例

论文中有两类主要算例。

### 3.1 一维 Marshak wave 精度测试

作者先用一维 Marshak wave 问题说明：**如果不在每个时间步内真正收敛非线性，即使时间格式看起来是二阶，也不一定能得到二阶精度。**

一维区域为 $x\in[0,1]$，两端给定混合通量边界：

$$
\frac14E+\frac12D_0\frac{\partial E}{\partial x}=\frac14\times 10^4,
$$

$$
\frac14E+\frac12D_1\frac{\partial E}{\partial x}=\frac14.
$$

该算例取

$$
D(T)=T^3,
$$

使用 flux limiter，并取 $\alpha=0$，初值为

$$
E(x,0)=1.
$$

比较的方法包括：

- SI1：一阶半隐式方法；
- SI2：二阶半隐式方法；
- NK1：一阶 Newton--Krylov；
- NK2：二阶 implicit midpoint Newton--Krylov。

结论是：非线性收敛的 Newton--Krylov 方法明显更准确，NK2 才真正接近二阶时间精度；而半隐式方法即使形式上使用二阶离散，也会因为没有收敛非线性而表现得接近一阶。

### 3.2 二维多材料算例

论文的主要性能测试使用二维多材料辐射扩散问题。区域为

$$
[0,1]\times[0,1],
$$

初值为 $E=1$，左右边界给定入射通量，上下为对称边界。材料拓扑由 $Z=10,20,50,100$ 的区域组成。

作者设计了三个模型：

| 模型 | 参数设置 | 难度来源 |
|---|---|---|
| M1 | $\alpha=1$，$D(E)=E^{1/4}=T$ | 非线性较弱 |
| M2 | $\alpha=0$，$C_v=1$，$D(E)=E^{3/4}=T^3$ | 存储项和扩散项都强非线性 |
| M3 | $\alpha=1$，$D(E)=E^{3/4}=T^3$，并使用 Wilson flux limiter | 强扩散非线性 + flux limiter |

因此，M1 是相对容易的问题，M2 和 M3 是用来检验算法鲁棒性的困难问题。

---

## 4. 算法总体框架

论文中的算法可以分成三层：

1. **最外层：时间推进**

   从 $t^n$ 推进到 $t^{n+1}$，并根据 $\eta$ 动态调整 $\Delta t$。

2. **中间层：非线性求解**

   每个时间步内要求解

   $$
   F(E^{n+1})=0.
   $$

   论文比较了 Picard 迭代和 Newton--Krylov 迭代。

3. **内层：线性 / Krylov 求解**

   Newton 线性子问题用 GMRES 求解；GMRES 的预条件器来自 Picard 线性化；Picard 预条件系统由 multigrid V-cycle 近似求解。

所以真正的关系是：

$$
\boxed{\text{multigrid 不是直接求原非线性方程，而是作为 Picard 预条件器的近似逆。}}
$$

---

## 5. 离散方法

### 5.1 空间离散：五点有限体积 / 有限差分扩散算子

在均匀网格上，扩散项采用五点格式：

$$
\nabla\cdot(D\nabla E)_{i,j}
\approx
\frac{1}{\Delta x}
\left[
D_{i+1/2,j}\frac{E_{i+1,j}-E_{i,j}}{\Delta x}
-
D_{i-1/2,j}\frac{E_{i,j}-E_{i-1,j}}{\Delta x}
\right]
$$

$$
+
\frac{1}{\Delta y}
\left[
D_{i,j+1/2}\frac{E_{i,j+1}-E_{i,j}}{\Delta y}
-
D_{i,j-1/2}\frac{E_{i,j}-E_{i,j-1}}{\Delta y}
\right].
$$

这可以理解为先在每个 face 上计算通量：

$$
F_{i+1/2,j}=-D_{i+1/2,j}\frac{E_{i+1,j}-E_{i,j}}{\Delta x},
$$

再对通量做散度。

### 5.2 face diffusion coefficient

因为扩散系数在材料界面处可能有强跳跃，所以 face 上的扩散系数不是简单算术平均，而是调和平均：

$$
D_{i+1/2,j}=\frac{2D_{i,j}D_{i+1,j}}{D_{i,j}+D_{i+1,j}}.
$$

这对多材料扩散问题很重要，因为调和平均更符合串联介质中通量连续的物理性质。

如果使用 Wilson limiter，则在 face 上进一步修正：

$$
D^L_{i+1/2,j}
=
\frac{1}{
\frac{1}{D_{i+1/2,j}}
+
\frac{|E_{i+1,j}-E_{i,j}|}{\frac12(E_{i,j}+E_{i+1,j})}
}.
$$

实现时应优先把 limiter 写成 **face-based** 形式，而不是只在 cell center 上处理，否则很容易与论文离散不一致。

### 5.3 时间离散

论文讨论了一阶和二阶时间格式。

一阶 backward Euler 可以写成

$$
F(E^{n+1})=
\frac{E^{n+1}-E^n}{\Delta t}
-
\nabla\cdot(D(T^{n+1})\nabla E^{n+1})=0.
$$

二阶 implicit midpoint rule 可以写成

$$
F(E^{n+1})=
\frac{E^{n+1}-E^n}{\Delta t}
-
\nabla\cdot\left(
D\left(\frac{T^n+T^{n+1}}{2}\right)
\nabla\left(\frac{E^n+E^{n+1}}{2}\right)
\right)=0.
$$

如果考虑一般的 $Q(E)$，更适合在代码中写成

$$
F(E^{n+1})=
\frac{Q(E^{n+1})-Q(E^n)}{\Delta t}
-
L(E^n,E^{n+1})=0,
$$

其中 $L$ 表示离散扩散算子。对于 M2，尤其要注意时间项是 $Q(E)$ 的差分，而不是简单地写成某个冻结质量系数乘以 $E^{n+1}-E^n$。

---

## 6. 每一步具体怎么求解

### 6.1 半隐式方法：作为基准方法

半隐式方法冻结旧时间层的扩散系数，求解

$$
\delta E-\Delta t\nabla\cdot(D(T^n)\nabla\delta E)
=
\Delta t\nabla\cdot(D(T^n)\nabla E^n),
$$

然后更新

$$
E^{n+1}=E^n+\delta E.
$$

这个方法线性稳定，但是没有在时间步内真正求解非线性方程，所以大时间步下精度会明显下降。

### 6.2 Picard 迭代：冻结系数，反复求线性问题

Picard 方法在第 $k$ 次非线性迭代时，把扩散系数冻结在当前近似 $E^{n+1,k}$ 上，求解一个线性修正方程：

$$
A(E^{n+1,k})\delta E^{k+1}=-F(E^{n+1,k}).
$$

然后阻尼更新：

$$
E^{n+1,k+1}=E^{n+1,k}+\xi\delta E^{k+1}.
$$

论文中阻尼因子取类似

$$
\xi=\min\left(1,\frac{1}{\|\delta E/E\|}\right),
$$

用于增强非线性迭代初期的稳定性。

Picard 的优点是线性系统相对简单，而且 Picard 线性化矩阵通常是对称正定的；缺点是面对强非线性时不够鲁棒，论文中 M2 模型下 Picard 甚至不能在给定非线性迭代次数内收敛。

### 6.3 Newton--Krylov：真正求解非线性方程

Newton 方法在第 $k$ 次非线性迭代中求解

$$
J(E^{n+1,k})\delta E^{k+1}=-F(E^{n+1,k}),
$$

然后更新

$$
E^{n+1,k+1}=E^{n+1,k}+\xi\delta E^{k+1}.
$$

这里 $J$ 是非线性残差 $F$ 对 $E$ 的 Jacobian。

但是论文没有显式形成 $J$，而是使用 Jacobian-free 近似：

$$
Jv\approx\frac{F(E+\varepsilon v)-F(E)}{\varepsilon}.
$$

也就是说，GMRES 只需要知道如何计算 $Jv$，不需要真的组装 Jacobian 矩阵。

### 6.4 右预条件形式

直接用 GMRES 解 Newton 方程往往效率不够，所以必须预条件。论文使用右预条件：

$$
(J\widetilde{M}^{-1})(\widetilde{M}\delta E)=-F(E).
$$

其中：

- $M$ 是 Picard 线性化得到的线性系统矩阵；
- $\widetilde{M}^{-1}$ 是 $M^{-1}$ 的近似；
- $\widetilde{M}^{-1}v$ 不是精确求解，而是用一个 multigrid V-cycle 近似得到。

具体到一次 GMRES 矩阵向量乘法：

1. 给定 Krylov 向量 $v$；
2. 用一个 multigrid V-cycle 近似求解

   $$
   My=v,
   $$

   得到预条件后的向量

   $$
   y\approx M^{-1}v;
   $$

3. 用 Jacobian-free 方式计算

   $$
   Jy\approx\frac{F(E+\varepsilon y)-F(E)}{\varepsilon}.
   $$

这一步就是论文算法的核心。

---

## 7. 多重网格是什么，以及论文如何使用多重网格

### 7.1 多重网格的基本思想

多重网格用于求解类似

$$
Au=f
$$

的大规模椭圆型线性系统。普通迭代方法比如 Jacobi 或 Gauss--Seidel 对高频误差消除较快，但对低频误差消除很慢。多重网格的思想是：

- 在细网格上做几步 smoothing，快速消除高频误差；
- 把残差限制到粗网格上；
- 在粗网格上，原本的低频误差会变成相对高频的误差，更容易被消除；
- 再把粗网格修正 prolongation 回细网格；
- 最后再做几步 post-smoothing。

### 7.2 一个 V-cycle 的基本流程

对线性系统

$$
Au=f,
$$

一个 V-cycle 可以写成：

1. 细网格预光滑：

   $$
   u\leftarrow \text{smoother}(A,u,f).
   $$

2. 计算残差：

   $$
   r=f-Au.
   $$

3. 限制到粗网格：

   $$
   r_c=Rr.
   $$

4. 构造 / 使用粗网格算子：

   $$
   A_c e_c=r_c.
   $$

5. 在粗网格递归求误差近似 $e_c$。

6. 延拓回细网格并修正：

   $$
   u\leftarrow u+Pe_c.
   $$

7. 细网格后光滑。

### 7.3 论文中的多重网格特点

论文的多重网格不是复杂的高阶 AMG，而是比较朴素、鲁棒的几何多重网格思想：

- 网格层级来自规则网格粗化；
- 层间转移使用简单的 piecewise constant transfer；
- smoother 使用 Jacobi 或 Gauss--Seidel 这类点迭代；
- 粗网格扩散系数用控制体积思想从细网格构造有效粗网格系数；
- 多重网格主要作为 Krylov 方法的预条件器，而不是单独作为最终求解器。

这点很关键：

$$
\boxed{\text{论文并不依赖 multigrid 自己完全鲁棒，而是让 GMRES 弥补 multigrid 的鲁棒性不足。}}
$$

### 7.4 为什么不用真实 Jacobian 做多重网格

真实 Jacobian 由于扩散系数依赖 $E$、温度前沿陡峭、flux limiter 等因素，可能是非对称甚至不定的。这样会让多重网格变得复杂且不稳定。

论文的策略是：

- Newton--Krylov 的矩阵向量乘法中体现真实 Jacobian；
- 预条件器中只使用 Picard 线性化矩阵；
- Picard 矩阵虽然只是近似，但更接近对称正定扩散系统，适合多重网格处理。

因此它牺牲了预条件器的精确性，换来了实现简单性和鲁棒性。

---

## 8. 可复现的算法伪代码

### 8.1 时间推进框架

```text
初始化 E = E0, t = 0, dt = dt0
while t < t_final:
    根据当前 dt 尝试从 E^n 求 E^{n+1}
    用 Newton--Krylov 求解 F(E^{n+1}) = 0
    计算 eta = max |E^{n+1}-E^n| / (E^{n+1}+E_floor)
    如果 eta 过大：拒绝该时间步，减小 dt 重算
    如果 eta 合适：接受时间步，t <- t + dt, E^n <- E^{n+1}
    下一步 dt 按 eta 目标调整，但增长不超过 10%
```

### 8.2 Newton--Krylov + multigrid preconditioning

```text
给定 E_old, dt
取初值 E = E_old
for nonlinear iteration k = 0,1,2,...:
    计算非线性残差 r = -F(E)
    如果 ||F(E)|| < nonlinear_tol：收敛，退出

    用 GMRES 近似求解 J(E) delta = r
    GMRES 中每次需要矩阵向量乘法时：
        输入 Krylov 向量 v
        用一个 multigrid V-cycle 近似求解 M y = v
        用差分近似 w = [F(E + eps*y) - F(E)] / eps
        返回 w

    得到 delta 后，做阻尼更新：
        E <- E + xi * delta
```

### 8.3 Picard 预条件器中的 $M$

$M$ 来自把非线性扩散系数冻结在当前 Newton 迭代状态 $E^k$ 上后得到的线性扩散问题。例如对一阶格式，可近似理解为

$$
M\delta E
\approx
\frac{\partial Q}{\partial E}(E^k)\frac{\delta E}{\Delta t}
-
\nabla\cdot\left(D(E^k)\nabla\delta E\right).
$$

对二阶 midpoint 格式，还需要对应地冻结 midpoint 状态下的扩散系数，并保持时间项、边界项与残差 $F$ 的定义一致。

---

## 9. 如果自己实现，需要规划哪些模块

为了尽量复现论文，代码不应写成一个大脚本，而应拆成下面这些模块。

### 9.1 `config.py`：参数与配置

负责保存所有参数：

- 网格大小：`nx, ny`；
- 计算区域：`xmin, xmax, ymin, ymax`；
- 模型：`M1, M2, M3`；
- 时间参数：`t_final, dt_initial, eta_target, E_floor`；
- 非线性参数：`nonlinear_tol, max_nonlinear_iter`；
- 线性参数：`linear_tol_factor, max_linear_iter`；
- 多重网格参数：层数、smoother 类型、前后光滑次数等。

建议用 `dataclass` 管理，避免到处传散乱参数。

### 9.2 `grid.py`：网格与索引

负责：

- 构造 cell-centered 网格；
- 保存 `dx, dy`；
- 管理二维数组形状；
- 必要时提供 flatten / unflatten，用于 GMRES 向量接口。

### 9.3 `geometry.py`：材料区域

负责根据论文二维算例生成 $Z(x,y)$：

```text
if x <= 0.5 and y <= 0.5: Z = 20
elif x >= 0.75 and y <= 0.25: Z = 100
elif sqrt((x-0.75)^2 + (y-0.75)^2) <= 0.15: Z = 50
else: Z = 10
```

该模块的输出应该是一个与网格同形状的 `Z` 数组。

### 9.4 `physics.py`：物理关系

负责实现：

- `temperature(E) = E**0.25`；
- `storage_quantity(E, model)`，即 $Q(E)$；
- `d_storage_dE(E, model)`，即 $Q'(E)$；
- `raw_diffusion(E, Z, model)`；
- M1、M2、M3 的扩散系数；
- Wilson limiter 的基础函数。

尤其要注意：M2 的时间项应基于

$$
Q(E)=E^{1/4},
$$

而不是简单使用 $E$ 本身。

### 9.5 `boundary.py`：边界条件

负责实现：

- 左右 Milne / Robin 边界；
- 上下对称边界；
- 边界 face diffusion；
- 边界通量对残差、Picard 线性化和 multigrid 预条件器的一致贡献。

这个模块非常容易出错。复现时要保证：

$$
\text{非线性残差、Picard 线性系统、多重网格 operator 使用同一套边界逻辑。}
$$

否则 Newton 残差和预条件器会求解两个不一致的问题。

### 9.6 `discretization.py`：离散算子与残差

负责：

- 计算 cell-centered diffusion；
- 计算 face diffusion；
- 做 harmonic mean；
- 对 M3 做 face-based Wilson limiter；
- 计算

  $$
  L(E)=\nabla\cdot(D(E)\nabla E);
  $$

- 构造非线性残差

  $$
  F(E^{n+1})=
  \frac{Q(E^{n+1})-Q(E^n)}{\Delta t}-L(E^n,E^{n+1}).
  $$

建议把一阶 BE 和二阶 midpoint 写成两个残差函数，或者用参数选择。

### 9.7 `picard_operator.py`：Picard 线性化矩阵 / 矩阵向量乘法

负责构造或应用

$$
M\delta E.
$$

可采用两种实现方式：

1. 显式组装稀疏矩阵，方便调试；
2. matrix-free matvec，方便与 multigrid 结合。

初期复现建议先显式组装矩阵，确认符号、边界和残差一致后，再优化成 matrix-free。

### 9.8 `multigrid.py`：多重网格 V-cycle

负责：

- 构造多层网格；
- restriction；
- prolongation；
- smoother，例如 weighted Jacobi / Gauss--Seidel；
- coarse grid operator；
- coarse-grid effective diffusion；
- 实现 `v_cycle(level, rhs, x0)`。

在论文框架中，`multigrid.py` 最重要的接口是：

```python
 y = v_cycle_for_picard_system(v)
```

也就是近似计算

$$
y\approx M^{-1}v.
$$

### 9.9 `jfnk.py`：Jacobian-free Newton--Krylov

负责实现：

- 非线性残差范数判断；
- finite-difference Jacobian-vector product：

  $$
  Jv\approx\frac{F(E+\varepsilon v)-F(E)}{\varepsilon};
  $$

- 右预条件 GMRES 接口；
- damping / line search；
- inexact Newton 线性容差：例如线性残差容差取当前非线性残差的 $10^{-2}$ 倍；
- 返回每个时间步的非线性迭代数和线性迭代数。

### 9.10 `time_stepper.py`：时间步控制

负责：

- 根据 $\eta$ 接受或拒绝时间步；
- 更新 $\Delta t$；
- 限制时间步增长不超过 $10\%$；
- 保存历史数据。

核心函数是：

$$
\eta=\max_{i,j}\frac{|E_{i,j}^{n+1}-E_{i,j}^n|}{E_{i,j}^{n+1}+E_{floor}}.
$$

### 9.11 `experiments.py`：论文算例驱动

负责运行论文中的组合：

- M1, M2, M3；
- $\eta=0.10,0.50$；
- 网格 $32^2,64^2,128^2,256^2$；
- 方法 SI1、Picard、NK2 等。

输出：

- 是否收敛；
- 平均线性迭代数；
- 平均非线性迭代数；
- 最大 $\eta$；
- 最终残差；
- CPU 时间；
- scaling exponent。

### 9.12 `diagnostics.py` 与 `plotting.py`

负责：

- 画温度等值线图；
- 画线性迭代数随时间变化；
- 画非线性迭代数随时间变化；
- 拟合 scaling exponent：

  $$
  \text{work}\sim N^s.
  $$

其中 $N=nx\times ny$。

### 9.13 `tests.py`：单元测试

建议至少测试：

1. 常数场下内部扩散残差应为零；
2. harmonic mean 是否正确；
3. Robin 边界通量符号是否正确；
4. M1、M2、M3 的 $D(E)$ 是否符合定义；
5. M3 limiter 是否真的作用在 face 上；
6. `residual`、`picard_operator`、`multigrid_operator` 的边界处理是否一致；
7. 对小网格，显式矩阵乘法和 matrix-free matvec 是否一致；
8. Newton 残差是否随迭代下降。

---

## 10. 复现时最容易出错的地方

### 10.1 把 multigrid 当成直接求解原非线性方程

论文中的 multigrid 不是直接解

$$
F(E)=0,
$$

而是近似解 Picard 预条件系统

$$
My=v.
$$

真正控制非线性收敛的是 Newton--Krylov。

### 10.2 M2 的时间项写错

M2 中 $\alpha=0$，所以

$$
Q(E)=E^{1/4}.
$$

因此时间项应是

$$
\frac{E_{new}^{1/4}-E_{old}^{1/4}}{\Delta t},
$$

或者在 Picard / Newton 线性化中使用对应的导数

$$
Q'(E)=\frac14E^{-3/4}.
$$

如果简单写成

$$
\frac{E_{new}-E_{old}}{\Delta t},
$$

就不再是论文的 M2 模型。

### 10.3 Wilson limiter 没有作用到 face

论文的 limiter 是 face diffusion coefficient 的修正。也就是说，它应该作用在 $D_{i+1/2,j}$、$D_{i,j+1/2}$ 这类面扩散系数上，而不是只修改 cell-centered $D_{i,j}$。

### 10.4 边界条件在残差和预条件器中不一致

如果非线性残差使用一套边界通量，而 Picard 矩阵或 multigrid 使用另一套边界通量，那么 GMRES 的预条件器就会和真实问题不匹配，表现为迭代数异常增加，甚至不收敛。

### 10.5 过度追求精确求解 Picard 线性系统

论文中 multigrid 的作用是给出便宜的近似逆。每次预条件通常只需要一个 V-cycle。线性系统不必过度求精，因为外层 GMRES 和 Newton 会继续校正。

---

## 11. 论文主要结果与理解

### 11.1 线性求解器层面

如果固定 Fourier number，多重网格接近理论上的线性标度；但是如果使用更真实的 $\eta$ 时间步控制，网格加密后有效 Fourier number 会变化，导致单独看线性系统时，多重网格不一定保持理想线性标度。

这说明：

$$
\text{线性求解器在理想测试中表现好，不代表在真实时间步控制下仍然完美。}
$$

### 11.2 非线性求解器层面

Picard 方法在弱非线性问题 M1 上还能工作，但在强非线性 M2 上不够鲁棒，在 M3 上迭代数也会随网格和时间变化明显波动。

Newton--Krylov 方法则能成功求解 M1、M2、M3，而且非线性迭代次数几乎不随网格显著恶化。

### 11.3 精度层面

论文强调：要得到时间离散的设计精度，必须在每个时间步内收敛非线性。半隐式方法虽然线性稳定，但因为没有真正收敛非线性，大时间步下会损失精度。二阶 Newton--Krylov 才能真正接近二阶时间精度。

---

## 12. 对自己代码复现的建议路线

如果从零开始或继续修改已有代码，建议按下面顺序推进。

### 第一阶段：先复现 PDE 和离散

1. 实现 M1，不开 limiter；
2. 实现二维材料分布 $Z(x,y)$；
3. 实现 Robin / symmetry 边界；
4. 实现 residual；
5. 用很小时间步检查解是否合理。

### 第二阶段：实现 Picard 线性化

1. 构造 Picard 矩阵；
2. 用直接稀疏求解器或 SciPy CG/GMRES 先验证；
3. 检查残差下降；
4. 再加入 M2 的 $Q(E)$ 时间项。

### 第三阶段：实现 Newton--Krylov

1. 写 JFNK 的有限差分 $Jv$；
2. 用无预条件 GMRES 在小网格上测试；
3. 加入 Picard 预条件；
4. 确认 M1、M2、M3 都能收敛。

### 第四阶段：实现 multigrid V-cycle

1. 先对常系数 Poisson / diffusion 测试 V-cycle；
2. 再加入变系数 diffusion；
3. 再加入材料跳跃；
4. 最后把 V-cycle 接到 Picard 预条件器中。

### 第五阶段：做论文表格和图的复现实验

1. 运行 $32^2,64^2,128^2$；
2. 记录线性 / 非线性平均迭代数；
3. 测试 $\eta=0.10$ 和 $\eta=0.50$；
4. 对比 Picard 和 Newton--Krylov 的鲁棒性；
5. 计算 scaling exponent。

---

## 13. 可直接引用的总结句

- 本文求解的是多材料平衡辐射扩散方程，其核心未知量是辐射能量密度 $E$，材料温度由平衡关系 $E=aT^4$ 给出。
- 方程的主要困难来自三个方面：辐射能量与温度的四次方关系、opacity 的温度依赖、多材料导致的扩散系数跳跃，以及 flux limiter 引入的梯度依赖。
- 论文的 Newton--Krylov 方法不显式形成 Jacobian，而是通过非线性残差差分近似 Jacobian-vector product。
- Picard 线性化本身作为非线性求解器并不总是鲁棒，但它非常适合作为 Newton--Krylov 的预条件器。
- 多重网格在本文中不是直接求解非线性方程，而是作为 Picard 预条件矩阵近似逆的一次 V-cycle。
- 对强非线性辐射扩散问题，是否在时间步内收敛非线性，直接决定方法能否达到设计时间精度。

---

## 14. 参考文献

Rider, W. J., Knoll, D. A., & Olson, G. L. (1999). *A Multigrid Newton--Krylov Method for Multimaterial Equilibrium Radiation Diffusion*. Journal of Computational Physics, 152, 164--191.
