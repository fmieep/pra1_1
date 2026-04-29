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


### 4.1 当前代码框架的阶段性目标：优先实现第二个二维多材料实验

结合当前这一组代码，现阶段不建议把目标写成“完整复现论文所有实验”。更合理的目标是：**先基本实现论文第二个实验，即二维多材料平衡辐射扩散算例**。也就是说，代码优先服务于第 3.2 节中的二维区域、多材料分布、左右 Milne / Robin 边界、上下对称边界，以及 M1、M2、M3 三个模型的 Picard / NK2 对比。

因此，当前代码的实现主线可以概括为：

$$
\boxed{
\text{二维多材料问题定义}
\rightarrow
\text{物理模型 M1/M2/M3}
\rightarrow
\text{face-based 离散}
\rightarrow
\text{midpoint 残差}
\rightarrow
\text{Picard 线性化}
\rightarrow
\text{MG 预条件 GMRES}
\rightarrow
\text{NK2 时间推进}
}
$$

其中，代码目前更接近论文中的 Tables XIII--XVI 和 Figures 10--11 的复现路径，而不是一维 Marshak wave 精度测试。换句话说：

- 第一阶段不要求先复现一维 SI1、SI2、NK1、NK2 的时间精度图；
- 第一阶段也不要求完整复现所有线性求解器比较，例如 SJCG、CJCG、MGCG；
- 重点应放在二维多材料算例中，检查 `picard` 与 `nk2` 在 M1、M2、M3 上的收敛性、平均线性迭代数、平均非线性迭代数和 scaling exponent。

从论文复现角度看，当前代码可以先回答下面几个问题：

1. 二维多材料几何、边界条件和初值是否已经按论文搭好；
2. M1、M2、M3 的存储项和扩散系数是否与论文一致；
3. M3 的 Wilson limiter 是否真正作用在 face 上；
4. NK2 是否在每个时间步内求解 midpoint 非线性残差；
5. GMRES 是否通过 Picard 型 multigrid V-cycle 做右预条件；
6. 在 $32^2,64^2,128^2,256^2$ 网格上，平均迭代数是否能形成类似论文表格的输出。

这一定位很重要，因为它能避免代码框架过早膨胀。当前代码不需要马上拆成十几个完全独立的工程模块，而是应该先围绕“二维实验能跑通、结果能统计、关键离散不跑偏”来组织。

---

## 5. 离散方法

### 5.1 空间离散 五点通量差分算子

严格说，论文在正文中把该离散写成 **five-point Laplacian**，也就是五点扩散算子；但它的写法是先把扩散系数放到单元界面 $i\pm1/2,j$、$i,j\pm1/2$，再用左右界面通量差构造散度。因此在均匀网格上，它可以理解为一种 **cell-centered finite-volume / flux-difference** 离散。更准确的表述是：

$$
\boxed{\text{论文采用的是带有限体积思想的单元中心五点通量差分离散，而不是高阶有限元或谱方法。}}
$$

所以在代码说明中，可以说“空间离散采用有限体积思想的 face-flux 形式”，但最好不要简单写成“全文使用有限体积法”而不加解释。

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

## 9. 按师兄建议拆分技术模块：便于逐步实现和单元测试

师兄评论中“技术组合太多”的意思是：当前说明把 Picard、Newton--Krylov、GMRES、多重网格、有限体积离散、边界条件、时间步控制等内容放在一条长链条里讲，整体上是对的，但不利于逐个排错。更适合代码复现的写法，是把每个技术点拆成可以单独实现、单独测试、单独替换的模块。

因此，当前实现可以按下面的技术模块重新组织：

| 技术模块 | 独立目标 | 主要检查点 |
|---|---|---|
| 物理模型模块 | 实现 $T(E)$、$Q(E)$、$Q'(E)$、$D(E,Z)$、Wilson limiter | M1/M2/M3 公式是否正确，正值保护是否合理 |
| 有限体积 / face-flux 离散模块 | 在固定 $D$ 下实现 $\nabla\cdot(D\nabla E)$ | 常数场扩散项应为 0，face diffusion 维度应正确，材料界面使用调和平均 |
| 边界条件模块 | 单独实现左右 Milne / Robin 边界和上下对称边界 | 左右边界通量符号、齐次边界项和非齐次边界项是否一致 |
| 时间离散模块 | 实现 BE / midpoint 残差和 $\eta$ 步长控制 | M2 时间项必须用 $Q(E^{n+1})-Q(E^n)$，而不是简单 $E^{n+1}-E^n$ |
| Picard 迭代模块 | 冻结系数，构造 Picard 线性修正问题 | Picard 矩阵与残差边界逻辑是否一致，阻尼更新是否保持 $E>0$ |
| 线性求解器模块 | 实现 GMRES / CG 等基础迭代求解器 | 对简单 SPD 扩散系统能否收敛，残差是否单调下降或整体下降 |
| 多重网格模块 | 实现 smoother、restriction、prolongation、V-cycle | 单独测试 $My=b$ 的残差下降，不先和 Newton 绑定 |
| Newton--Krylov 模块 | 实现 Jacobian-free $Jv$ 和右预条件 | $Jv$ 差分扰动是否合理，GMRES 中是否调用 $M^{-1}v$ |
| 实验与报告模块 | 批量运行 M1/M2/M3、不同网格和不同 $\eta$ | 自动输出收敛性、平均迭代数、失败原因和测试报告 |

这样拆分后，推荐的实现顺序是：

```text
1. 先测试物理函数：T(E), Q(E), D(E,Z), limiter
2. 再测试有限体积离散：固定 D 下的 face flux 和 divergence
3. 再测试边界条件：左/右 Robin 通量、上/下对称边界
4. 再测试时间残差：BE 或 midpoint residual
5. 再测试 Picard：不加 Newton，先看冻结系数迭代是否合理
6. 再测试 multigrid：单独解 Picard 线性系统 My=b
7. 再测试 GMRES + MG 预条件：确认线性层能正常下降
8. 最后接入 Newton--Krylov：检查 JFNK 外层非线性收敛
9. 最后做实验表格和测试报告
```

这也意味着，代码仓库中应有一个 `tests/` 文件夹。当前项目已经采用 pytest，并按技术模块拆成了下面这些测试文件：

```text
tests/
├── test_problem_physics.py
├── test_discretization.py
├── test_linear_multigrid.py
├── test_methods_driver.py
└── test_integration_experiments.py
```

其中 `test_integration_experiments.py` 不需要跑大网格，只在小网格上检查驱动器、统计函数和短时间运行是否正常。这样比直接跑完整论文表格更容易定位问题。

---

## 10. 结合当前代码的实现框架说明

这一节不再只写“理想情况下应该拆哪些模块”，而是结合当前已经上传的代码，说明它们如何共同服务于**第二个二维多材料实验**。同时，为了回应“每个技术都单独写”的建议，下面的说明不只按文件名介绍，也强调每个文件对应的独立技术职责。当前代码框架已经不只是一个理论规划，而是形成了一个可运行的最小复现主线：

$$
\texttt{test.py / experiments.py}
\rightarrow
\texttt{config.py}
\rightarrow
\texttt{problem.py}
\rightarrow
\texttt{driver.py}
\rightarrow
\texttt{methods.py}
\rightarrow
\texttt{discretization.py + physics.py + solvers.py}.
$$

当前项目还增加了两个辅助层：`generate_test_report.py` 用于生成中文测试报告，`compare_paper_results.py` 用于从 checkpoint 中读取 NK2 结果并生成论文三线表对比报告。

更直观地说：

$$
\boxed{
\text{实验配置}
\rightarrow
\text{二维问题搭建}
\rightarrow
\text{时间推进}
\rightarrow
\text{非线性残差}
\rightarrow
\text{GMRES 求 Newton/Picard 修正}
\rightarrow
\text{multigrid 近似 Picard 预条件逆}
}
$$

### 10.1 `config.py`：控制实验参数和求解器参数

`config.py` 是当前代码的参数入口，主要包含两个 dataclass：

- `SolverConfig`：控制非线性求解器和线性求解器，例如 `method="picard"` 或 `method="nk2"`、非线性容差、GMRES 最大迭代次数、是否使用 multigrid 预条件、JFNK 差分扰动模式、阻尼模式等；
- `RunConfig`：控制具体算例，例如模型 `M1/M2/M3`、网格大小、终止时间、目标 `eta`、初始时间步、时间步增长限制和 `E_floor`。

当前 `default_run_config` 已经根据论文二维实验的不同模型设置了不同终止时间：M1 到 $t=5.0$，M2 到 $t=0.005$，M3 根据 `eta` 取 $t=0.1$ 或 $t=0.5$。这说明代码目标已经明显偏向论文二维多材料算例，而不是一维 Marshak wave 测试。

因此，`config.py` 的定位可以写成：

$$
\boxed{\texttt{config.py} = \text{二维多材料实验的总参数层}}
$$

### 10.2 `problem.py`：搭建二维多材料算例

`problem.py` 对应论文第二个实验的问题定义。它完成四件事：

1. 用 `build_grid` 构造 $[0,1]\times[0,1]$ 上的 cell-centered 均匀网格；
2. 用 `build_material_map` 生成论文中的 $Z=10,20,50,100$ 多材料分布；
3. 用 `build_boundary_config` 设置左右 Milne / Robin 边界和上下对称边界；
4. 用 `build_initial_energy` 设置初值 $E=1$。

所以 `problem.py` 对应的是论文中“二维多材料几何 + 初值 + 边界条件”这一层。当前阶段应优先保证这里完全符合论文第二个实验，因为后面的求解器是否正确，首先取决于问题定义是否正确。

可以把这一层概括为：

$$
\boxed{\texttt{problem.py} = \text{二维多材料物理算例生成器}}
$$

### 10.3 `physics.py`：实现 M1、M2、M3 的物理关系

`physics.py` 负责把论文中的物理模型转成代码函数，主要包括：

- `energy_to_temperature(E)`：实现 $T=E^{1/4}$；
- `storage_quantity(E, model)`：实现时间项中的 $Q(E)$；
- `mass_coefficient(E, model)`：实现 Picard 线性化和预条件器中的 $Q'(E)$；
- `raw_diffusion_coefficient(E, Z, model)`：实现 M1、M2、M3 的原始扩散系数；
- `wilson_limiter(...)`：实现 M3 的 Wilson flux limiter。

其中最关键的是 M2：

$$
Q(E)=E^{1/4},
\qquad
Q'(E)=\frac14E^{-3/4}.
$$

这意味着 M2 的时间项不能写成简单的 $(E^{n+1}-E^n)/\Delta t$。当前代码把 `storage_quantity` 和 `mass_coefficient` 单独放在 `physics.py` 中，是比较合理的，因为这能避免在残差函数、Picard 线性化和 multigrid 预条件器里各写一套不一致的时间项。

可以把这一层概括为：

$$
\boxed{\texttt{physics.py} = \text{M1/M2/M3 的模型公式层}}
$$

### 10.4 `discretization.py`：实现 face-based 空间离散和边界通量

`discretization.py` 是二维实验复现中最核心、也最容易出错的模块。它负责把连续算子

$$
\nabla\cdot(D\nabla E)
$$

离散成 face flux 再求散度的形式。

当前代码中比较重要的设计包括：

1. `compute_face_diffusion` 使用调和平均把 cell-centered diffusion 放到 face 上；
2. `apply_wilson_limiter_on_faces` 对 M3 在 x-face 和 y-face 上分别使用 Wilson limiter；
3. `build_boundary_data` 和 `_milne_robin_coefficients` 把 Milne / Robin 边界条件离散成可复用的边界通量系数；
4. `build_full_boundary_flux` 用于非线性残差中的完整边界通量；
5. `build_homogeneous_boundary_flux` 用于 Picard 线性化和 multigrid 预条件器中的齐次线性边界作用；
6. `diffusion_operator_split` 支持 midpoint 格式中“扩散系数状态”和“梯度状态”不同的写法。

这里与论文最对应的一点是：当前代码没有只在 cell center 上做 Wilson limiter，而是把 limiter 写到了 face diffusion coefficient 上。这比简单的 cell-centered limiter 更接近论文形式。

可以把这一层概括为：

$$
\boxed{\texttt{discretization.py} = \text{二维 face-based 扩散算子与边界离散层}}
$$

### 10.5 `methods.py`：实现 Picard 和 NK2 的非线性求解框架

`methods.py` 是当前算法主干。它把论文的二阶 implicit midpoint Newton--Krylov 方法落实到了代码中。

当前最关键的函数是 `residual_nk2`，它实现的是：

$$
F(E^{n+1})=
\frac{Q(E^{n+1})-Q(E^n)}{\Delta t}
-
\nabla\cdot\left[
D\left(\frac{T^n+T^{n+1}}{2}\right)
\nabla\left(\frac{E^n+E^{n+1}}{2}\right)
\right].
$$

这正是当前代码想要复现第二个实验时最重要的非线性残差。

`methods.py` 中还包含：

- `build_midpoint_states`：构造 midpoint 格式需要的两个状态；
- `build_picard_linearization`：冻结当前状态，构造 Picard 型线性化算子 $M$；
- `picard_step`：用 Picard 外迭代推进一个时间步；
- `nk2_step`：用 Jacobian-free Newton--Krylov 推进一个时间步；
- `jacobian_free_matvec`：用有限差分近似 $Jv$；
- `_damped_update`：实现阻尼更新和正值保护。

从论文角度看，`methods.py` 对应的是：

$$
\boxed{
\text{Picard residual / Picard linearization / JFNK nonlinear step}
}
$$

也就是说，它是把“论文算法”真正连接到程序运行的地方。

### 10.6 `solvers.py`：GMRES 和 Picard 型 multigrid 预条件器

`solvers.py` 负责内层线性代数。当前代码中包含：

- restarted `gmres`；
- 一个保留接口的 `cg`；
- weighted Jacobi smoother；
- restriction 和 prolongation；
- 多重网格层级构造；
- 递归 `v_cycle`；
- `PicardMGPreconditioner`。

这与论文的关系是：GMRES 并不直接使用 multigrid 去解原始非线性方程，而是在每次 Krylov 迭代中调用

$$
y\approx M^{-1}v
$$

其中这个近似逆由一次 V-cycle 给出。当前代码中 `PicardMGPreconditioner.apply(v)` 正是这个角色。

因此，`solvers.py` 的定位不是“单独的 multigrid 求解器”，而是：

$$
\boxed{\texttt{solvers.py} = \text{GMRES + Picard 型 MG 右预条件器}}
$$

这也解释了为什么 multigrid 层级里使用的是 Picard 冻结系数矩阵，而不是真实 Newton Jacobian。

### 10.7 `driver.py`：时间推进和 `eta` 控制

`driver.py` 负责把单步非线性求解器包在时间循环外面。它主要完成：

1. 从 $t=0$ 开始循环推进；
2. 调用 `solve_one_step` 得到 $E^{n+1}$；
3. 计算

   $$
   \eta=\max_{i,j}\frac{|E_{i,j}^{n+1}-E_{i,j}^{n}|}{E_{i,j}^{n+1}+E_{floor}};
   $$

4. 根据目标 `eta_target` 更新下一步时间步长；
5. 记录线性迭代数、非线性迭代数、残差、时间步等历史信息。

这一层对应论文二维实验中的 realistic time step control。当前代码的 `compute_eta`、`update_dt` 和 `run_simulation` 已经具备基本实验统计功能。

可以把这一层概括为：

$$
\boxed{\texttt{driver.py} = \text{带 }\eta\text{ 控制的时间推进器}}
$$

### 10.8 `experiments.py`：批量复现实验表格

`experiments.py` 更接近论文表格复现层。它提供：

- `run_case`：运行一个指定的 `(method, model, eta, grid)` 组合；
- `run_table_xiii_to_xvi`：批量运行 `picard/nk2`、`M1/M2/M3`、`eta=0.10/0.50`、`32^2` 到 `256^2`；
- `fit_scaling_exponent`：拟合平均迭代数随自由度变化的幂次

  $$
  \text{avg\_iters}\sim N^p.
  $$

需要注意的是，论文表格中的 scaling exponent 定义为计算工作量

$$
\text{work}\sim N^s.
$$

因为每一次迭代本身需要 $O(N)$ 的工作量，所以若要和论文中的 $s$ 对齐，应使用

$$
s = 1 + p.
$$

当前项目中 `compare_paper_results.py` 已经按这个论文定义计算对比报告中的 Scaling exponent。

这一模块对应论文后半部分的平均迭代数表格，尤其接近 Picard 和 Newton--Krylov 对比的 Tables IX--XVI。

不过从当前阶段看，建议先不要一上来跑完整 `32,64,128,256` 的全组合。可以先用：

$$
\text{method} = \texttt{nk2},\quad
\text{model}=\texttt{M1,M2,M3},\quad
\eta=0.10,
\quad
\text{grid}=32^2.
$$

确认三组都能稳定完成后，再扩大到 `64`、`128`，最后再考虑 `256`。

### 10.9 `test.py`：当前最方便的实验总开关

`test.py` 是当前最适合日常调试的入口。它通过顶部变量控制：

- 单个测试还是批量测试；
- 使用 `nk2` 还是 `picard`；
- 运行 M1、M2、M3 哪个模型；
- 使用哪个 `eta`；
- 网格大小；
- GMRES、非线性容差、JFNK 扰动方式、阻尼方式等。

因此，在现阶段写论文复现说明时，可以把 `test.py` 描述为：

$$
\boxed{\texttt{test.py} = \text{面向调试的二维实验运行入口}}
$$

它的作用不是论文算法的一部分，而是方便快速切换实验组合，观察每个 case 是否收敛、平均线性迭代数是多少、平均非线性迭代数是多少。

### 10.10 当前代码框架与“第二个实验”的对应关系

可以用下面这张表总结当前代码与论文第二个实验之间的对应关系：

| 论文第二个实验要素 | 当前代码模块 | 当前实现重点 |
|---|---|---|
| 二维区域 $[0,1]^2$ | `problem.py` | cell-centered 均匀网格 |
| 多材料 $Z=10,20,50,100$ | `problem.py` | 生成材料分布图 |
| 初值 $E=1$ | `problem.py` | 初始化能量场 |
| 左右 Milne / Robin 边界 | `problem.py` + `discretization.py` | 边界通量系数和 ghost elimination |
| 上下对称边界 | `problem.py` + `discretization.py` | 零法向通量 |
| M1/M2/M3 | `physics.py` | $Q(E)$、$D(E)$、Wilson limiter |
| face diffusion | `discretization.py` | 调和平均和 face limiter |
| midpoint NK2 残差 | `methods.py` | `residual_nk2` |
| Picard 线性化 | `methods.py` | `build_picard_linearization` |
| GMRES | `solvers.py` | matrix-free Krylov 求解 |
| multigrid 预条件 | `solvers.py` | `PicardMGPreconditioner.apply` |
| 时间步控制 | `driver.py` | `eta` 控制和历史记录 |
| 表格型批量实验 | `experiments.py` / `test.py` | 平均迭代数和实验结果保存 |
| 论文结果对比 | `compare_paper_results.py` | 三线表对比、相对误差和论文定义的 scaling exponent |

### 10.11 当前框架下暂时不需要优先做的内容

为了先把第二个实验跑稳，下面这些内容可以暂时放到后面：

1. 一维 Marshak wave 精度测试；
2. SI1、SI2、NK1 的完整复现；
3. SJCG、CJCG、MGCG 等线性求解器的完整横向比较；
4. 与论文图 1、图 2 对应的时间精度阶数拟合；
5. 更复杂的可视化和论文级绘图；
6. 大规模性能优化。

现阶段更重要的是保证当前主线满足：

$$
\boxed{
\text{M1/M2/M3 能跑通}
+
\text{NK2 残差定义正确}
+
\text{MG 预条件器与 Picard 线性化一致}
+
\text{输出可对照论文表格}
}
$$

这比一开始就追求完整工程化更适合当前复现进度。

## 11. 复现时最容易出错的地方

### 11.1 把 multigrid 当成直接求解原非线性方程

论文中的 multigrid 不是直接解

$$
F(E)=0,
$$

而是近似解 Picard 预条件系统

$$
My=v.
$$

真正控制非线性收敛的是 Newton--Krylov。

### 11.2 M2 的时间项写错

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

### 11.3 Wilson limiter 没有作用到 face

论文的 limiter 是 face diffusion coefficient 的修正。也就是说，它应该作用在 $D_{i+1/2,j}$、$D_{i,j+1/2}$ 这类面扩散系数上，而不是只修改 cell-centered $D_{i,j}$。

### 11.4 边界条件在残差和预条件器中不一致

如果非线性残差使用一套边界通量，而 Picard 矩阵或 multigrid 使用另一套边界通量，那么 GMRES 的预条件器就会和真实问题不匹配，表现为迭代数异常增加，甚至不收敛。

### 11.5 过度追求精确求解 Picard 线性系统

论文中 multigrid 的作用是给出便宜的近似逆。每次预条件通常只需要一个 V-cycle。线性系统不必过度求精，因为外层 GMRES 和 Newton 会继续校正。

---

## 12. 论文主要结果与理解

### 12.1 线性求解器层面

如果固定 Fourier number，多重网格接近理论上的线性标度；但是如果使用更真实的 $\eta$ 时间步控制，网格加密后有效 Fourier number 会变化，导致单独看线性系统时，多重网格不一定保持理想线性标度。

这说明：

$$
\text{线性求解器在理想测试中表现好，不代表在真实时间步控制下仍然完美。}
$$

### 12.2 非线性求解器层面

Picard 方法在弱非线性问题 M1 上还能工作，但在强非线性 M2 上不够鲁棒，在 M3 上迭代数也会随网格和时间变化明显波动。

Newton--Krylov 方法则能成功求解 M1、M2、M3，而且非线性迭代次数几乎不随网格显著恶化。

### 12.3 精度层面

论文强调：要得到时间离散的设计精度，必须在每个时间步内收敛非线性。半隐式方法虽然线性稳定，但因为没有真正收敛非线性，大时间步下会损失精度。二阶 Newton--Krylov 才能真正接近二阶时间精度。

---



## 13. 面向测试报告的最小检查清单

当前代码已经有 pytest 测试和自动测试报告；如果后续继续扩展，可以仍然采用“从底层到顶层”的最小测试清单：

| 测试层级 | 建议测试内容 | 通过标准 |
|---|---|---|
| 单函数测试 | `energy_to_temperature`、`storage_quantity`、`harmonic_mean`、`compute_eta` | 对简单输入给出可手算结果 |
| 离散算子测试 | 常数场、线性场、固定扩散系数、材料跳跃界面 | 数组形状正确，常数场内部扩散项接近 0 |
| 边界测试 | 左右 Robin、上下 symmetric | 边界通量符号符合物理方向，齐次/非齐次项分离清楚 |
| 线性求解测试 | GMRES、multigrid V-cycle | 对小规模线性系统残差下降 |
| 非线性 smoke test | 小网格 M1/M2/M3 短时间运行 | 不崩溃，残差低于容差，迭代数被记录 |
| 实验报告测试 | 自动生成表格或 html 报告 | 报告中包含通过/失败数量、失败原因、平均迭代数 |

这份清单的目标不是一开始就证明完全复现论文，而是保证每一层技术模块都有独立证据支持。这样当最终 M2 或 M3 不收敛时，可以快速判断问题是出在物理模型、有限体积离散、边界通量、Picard 线性化、多重网格，还是 JFNK 外层。

---

## 14. 可直接引用的总结句

- 本文求解的是多材料平衡辐射扩散方程，其核心未知量是辐射能量密度 $E$，材料温度由平衡关系 $E=aT^4$ 给出。
- 方程的主要困难来自三个方面：辐射能量与温度的四次方关系、opacity 的温度依赖、多材料导致的扩散系数跳跃，以及 flux limiter 引入的梯度依赖。
- 论文的 Newton--Krylov 方法不显式形成 Jacobian，而是通过非线性残差差分近似 Jacobian-vector product。
- Picard 线性化本身作为非线性求解器并不总是鲁棒，但它非常适合作为 Newton--Krylov 的预条件器。
- 多重网格在本文中不是直接求解非线性方程，而是作为 Picard 预条件矩阵近似逆的一次 V-cycle。
- 对强非线性辐射扩散问题，是否在时间步内收敛非线性，直接决定方法能否达到设计时间精度。
