# SEM3D 反演算法解释

## 1. 一轮 inversion iteration 的总目标

一轮迭代的目标不是直接“让材料变好看”，而是从当前材料模型出发：

```text
当前材料模型
→ 正演波场
→ 与观测数据比较
→ 得到残差
→ 伴随传播
→ 计算材料梯度
→ 生成候选材料
→ 重新正演候选材料
→ 判断是否接受
```

如果加入 TV regularization，则材料更新方向同时考虑两件事：

1. 数据是否拟合得更好；
2. 材料空间分布是否出现没有物理依据的锯齿振荡。

---

## 2. Parent material

每一轮开始时都有一个当前被接受的材料模型：

\[
m_k = (\lambda_k,\mu_k,ho_k).
\]

其中：

- \(\lambda\)：Lamé 第一参数；
- \(\mu\)：剪切模量；
- \(ho\)：密度。

这些材料参数被写入 SEM3D 使用的 HDF5 文件：

```text
Mat_0_Kappa.h5
Mat_0_Mu.h5
Mat_0_Density.h5
```

SEM3D 实际使用的体积模量为：

\[
\kappa = \lambda + rac{2}{3}\mu.
\]

---

## 3. Strict forward

输入包括：

```text
mesh
material HDF5
source time function
source position
stations
input.spec
```

SEM3D 求解弹性波动方程，得到正演波场。

主要输出有两类：

### Receiver displacement

物理接收器位置上的位移：

\[
u_{\mathrm{sim}}(x_r,t).
\]

它用于和真实观测数据比较。

### DUDX

控制点上的位移空间导数：

\[

abla u.
\]

它不会直接用于计算 misfit，而是供后续梯度计算使用。

所以：

```text
UU / displacement
→ residual 与 objective

DUDX
→ gradient
```

---

## 4. Residual 与数据目标函数

在同一个 receiver、同一个分量、同一个时间采样上计算：

\[
r(x_r,t)
=
u_{\mathrm{sim}}(x_r,t)
-
u_{\mathrm{obs}}(x_r,t).
\]

数据目标函数为：

\[
J_{\mathrm{data}}
=
rac12
\sum_r
\int
\|r(x_r,t)\|^2\,dt.
\]

Residual 阶段本身不需要重新运行 SEM3D，它只是对 HDF5 trace 做：

```text
receiver 坐标匹配
时间轴检查
分量对齐
simulated - observed
时间反转
adjoint source HDF5
```

---

## 5. Adjoint simulations

Residual 时间反转后，作为伴随源放回 receiver 位置。

对于三个位移分量：

```text
x residual
y residual
z residual
```

分别运行伴随传播。

Full strict 中采用：

```text
x：10 batches
y：10 batches
z：10 batches
总计：30 batches
```

这里的 30 batches 不是把地下网格切成 30 块，而是把大量 receiver residual sources 分批放入 SEM3D。

每个伴随 batch 输出对应的伴随 DUDX。

---

## 6. Data RHS

在相同空间控制点和相同时间上，将：

```text
forward DUDX
×
adjoint DUDX
```

进行时间积分，得到：

\[
RHS_{\lambda,\mathrm{data}},
\qquad
RHS_{\mu,\mathrm{data}}.
\]

它们是离散后的材料灵敏度右端项，还不是最终材料更新方向。

三个位移分量的贡献最后相加：

\[
RHS_{\lambda,\mathrm{data}}
=
RHS_{\lambda,x}
+
RHS_{\lambda,y}
+
RHS_{\lambda,z},
\]

\[
RHS_{\mu,\mathrm{data}}
=
RHS_{\mu,x}
+
RHS_{\mu,y}
+
RHS_{\mu,z}.
\]

---

## 7. Mtilde solve

离散梯度不是直接等于 RHS，而是通过材料质量矩阵：

\[
\widetilde M g_\lambda
=
RHS_\lambda,
\]

\[
\widetilde M g_\mu
=
RHS_\mu.
\]

求解后得到：

\[
g_\lambda,
\qquad
g_\mu.
\]

这一步把离散 RHS 转换成与材料参数空间一致的梯度。

---

## 8. TV regularization

TV 的目标是减少材料中的锯齿振荡，同时允许明显界面存在。

平滑 TV 定义为：

\[
R_{\mathrm{TV}}(m)
=
\int_\Omega
\sqrt{
|
abla m|^2+\epsilon^2
}
\,d\Omega.
\]

其中 \(\epsilon\) 用于避免梯度为零时不可导。

TV 不是在候选材料生成后简单做滤波，而是在材料梯度形成阶段加入：

\[
RHS_{\lambda,\mathrm{total}}
=
RHS_{\lambda,\mathrm{data}}
+
lpha_\lambda
RHS_{\lambda,\mathrm{TV}},
\]

\[
RHS_{\mu,\mathrm{total}}
=
RHS_{\mu,\mathrm{data}}
+
lpha_\mu
RHS_{\mu,\mathrm{TV}}.
\]

然后仍然使用同一个 Mtilde 系统：

\[
\widetilde M g_{\lambda,\mathrm{total}}
=
RHS_{\lambda,\mathrm{total}},
\]

\[
\widetilde M g_{\mu,\mathrm{total}}
=
RHS_{\mu,\mathrm{total}}.
\]

因此 TV 被真正融入一轮 iteration，而不是独立脚本。

---

## 9. Candidate generation

根据总梯度和步长生成候选材料：

\[
m_{\mathrm{candidate}}
=
m_{\mathrm{parent}}
-
s\,d.
\]

代码会生成不同步长的 candidate，例如：

```text
0.10 MPa
0.25 MPa
0.50 MPa
1.00 MPa
```

并重新写出：

```text
Mat_0_Kappa.h5
Mat_0_Mu.h5
Mat_0_Density.h5
```

---

## 10. Candidate forward

Candidate forward 的作用是用候选材料重新求解波场。

只有这一步之后，才能得到候选模型真实的：

\[
J_{\mathrm{data,candidate}}.
\]

因此不运行 candidate forward 时，可以验证 TV 数学和代码集成，但不能声称新的候选模型在真实波场下已经下降。

---

## 11. Total objective

有 TV 时，接受判据不应只看数据 misfit，而应比较：

\[
J_{\mathrm{total}}
=
J_{\mathrm{data}}
+
lpha_\lambda
R_{\mathrm{TV}}(\widehat\lambda)
+
lpha_\mu
R_{\mathrm{TV}}(\widehat\mu).
\]

接受条件为：

\[
J_{\mathrm{total,candidate}}
<
J_{\mathrm{total,parent}}.
\]

通过后，candidate 才成为下一轮 accepted material。

---

## 12. 一轮 iteration 的完整逻辑

```text
parent material
→ strict forward
→ residual
→ 30 adjoint batches
→ data RHS
→ optional TV RHS
→ total RHS
→ Mtilde total gradient
→ candidate materials
→ candidate forward
→ J_total
→ accept or reject
→ next iteration
```

TV 的插入位置是：

```text
data RHS
→ TV RHS
→ total RHS
→ Mtilde
```

它不修改 SEM3D solver，只修改优化层中的材料更新方向。
