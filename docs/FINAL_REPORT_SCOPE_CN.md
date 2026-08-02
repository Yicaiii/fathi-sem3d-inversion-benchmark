# 最终汇报边界

## 已完成

### 1. 通用 iteration context

运行路径由 config 和 iteration context 决定，不再依赖固定用户目录或固定 iteration 编号。

### 2. Canonical runner

统一入口能够表达：

```text
gradient
regularization
tv_candidates
tv_acceptance_plan
```

TV 不再是独立于 iteration 的历史脚本，而是一轮反演中的可选阶段。

### 3. TV 数学实现

已完成：

```text
Q1 smoothed TV value
TV derivative
active-set restriction
data RHS + TV RHS
Mtilde total gradient
candidate generation
total-objective evaluation
```

### 4. 轻量验证

已通过：

```text
41 passed
4 subtests passed
```

其中包括：

```text
constant TV derivative test
directional finite-difference test
alpha=0 regression
synthetic non-homogeneous material test
synthetic candidate generation
total-objective acceptance test
```

并确认：

```text
SEM3D launched = False
accepted state mutated = False
```

### 5. Mini physical loop

已有 mini benchmark 验证了：

```text
forward
→ residual
→ adjoint
→ RHS
→ Mtilde
→ candidate forward
→ acceptance
```

因此原始 SEM3D 物理闭环已被实际运行验证。

---

## 本次没有重新运行

为了避免 full-scale 计算成本，本次 refactor 没有重新执行：

```text
full strict forward
30 full strict adjoint batches
non-zero-TV candidate forward
```

因此不能声称：

> 新的 full strict 非零 TV candidate 已经通过真实 SEM3D 波场证明 objective 下降。

---

## 可以准确汇报的结论

> TV regularization has been integrated into the generic inversion iteration between data-RHS assembly and the Mtilde solve. The implementation computes the TV value and derivative from the current parent material, restricts the TV contribution to the active control set, combines it with the data RHS, and solves for a total material gradient. Lightweight derivative, alpha-zero, synthetic candidate, and total-objective tests pass without launching SEM3D or mutating the accepted state.

中文：

> TV 正则化已经接入通用反演 iteration，位置位于 data RHS 组装之后和 Mtilde 求解之前。系统从当前 parent material 计算 TV 值和导数，将 TV contribution 限制到 active control set，与 data RHS 组合，并求解总材料梯度。方向导数、alpha=0 回归、合成候选材料和总目标函数测试均已通过，整个轻量验证过程没有启动 SEM3D，也没有修改 accepted state。

---

## 保留为后续重型实验

后续如有计算资源，可执行：

```text
non-zero TV weights
→ full total gradient
→ candidate materials
→ candidate forward SEM3D
→ candidate J_data
→ candidate J_total
→ physical acceptance
```

这属于结果验证，不属于当前代码集成是否完成的必要条件。
