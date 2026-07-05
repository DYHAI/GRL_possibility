# 论文记忆（同步自项目对话）

> 主仓库：https://github.com/DYHAI/GRL_possibility  
> 完整 LaTeX 片段见 `paper/latex/`；图表清单见 `paper/figures.md`

---

## 核心论点

**RMSE/MSE 最优的确定性预报 = 条件期望** \(\mathbb{E}[\mathbf{X}_{t+\Delta t}\mid \mathbf{X}_t]\)，不是完整转移分布 \(p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t)\)。

因此即使 bulk RMSE 很好，也会出现三类**结构性**问题（不是单纯数据或网络容量问题）：

1. **高频能谱耗散 / 过平滑（Over-smoothing）**  
   条件期望 = 对所有可能未来状态的加权平均；在频域上相当于低通滤波。混沌大气中多条分叉轨迹的高频相位被平均掉 → 预报场更平滑。  
   **KH 验证指标**：ρ 径向功率谱、high-k power ratio（forecast/truth < 1，随 rollout 步数恶化）。

2. **极端值低估（Underestimation of extremes）**  
   离散形式 \(g^\*(s^{(i)})=\sum_j P_{ij} s^{(j)}\)：长尾小概率大值被常态值拉向气候均值。  
   **KH 验证指标**：domain max ρ 的 peak ratio、tail exceedance P(ρ≥阈值)、high-tail bias（真值 top 5% 像素上预报−真值 < 0）。

3. **非线性物理不守恒（Violation of conservation / non-closure）**  
   Jensen 不等式：对凸非线性 \(f\)，\(\mathbb{E}[f(X)\mid\cdot]\neq f(\mathbb{E}[X\mid\cdot])\)。  
   条件期望场一般**不是**原 Euler/NS 方程的解，而是 \(\partial_t \bar u = \mathcal{N}(\bar u) + R\) 带闭合误差 \(R\neq 0\)。  
   **理论**：Proposition `prop:nonlinear_nonclosure` + Jensen 展开（见 `latex/sec02_jensen_nonclosure.tex`）。

---

## 论文结构（Section 路线图）

| Section | 内容 |
|---------|------|
| **2 理论** | Markov 表述 → MSE 训练 → 证明 MSE 最优 = 条件期望（Proposition + Corollary 加权平均）→ 三机制讨论 → Jensen 非闭合命题 |
| **3 Data & Methods** | Markov 玩具实验（K=3,10,100）；KH Trixi 200-member ensemble；U-Net 512×4 one-step |
| **4 Results & Discussion** | 两实验承接理论；重点写三机制（可先定性，数值后填） |
| **Conclusion** | 呼应 Key Points：RMSE 最优 ≠ 分布学习；信息熵/变异性损失 |

Introduction 末段 roadmap 见 `latex/intro_roadmap.tex`。

---

## 实验 1：Markov 转移矩阵

- **目的**：可控验证 Proposition——MSE 学标量条件均值，Cross-Entropy 学整行转移概率。
- **状态空间**：\(K\in\{3,10,100\}\)，\(s^{(j)}\) 在 \([-1,1]\) 均匀。
- **两模型**：MLP(RMSE) 预测下一态标量；MLP(CE) + softmax 预测整行 \(P_{ij}\)。
- **预期**：RMSE 相近；KL(P_true ‖ P_pred) 上 CE 远小于 RMSE 模型。
- **代码**：`python experiments/markov/markov_mlp.py --sizes 3 10 100`
- **重绘论文图**：`python experiments/markov/markov_mlp.py --replot-only --sizes 3 10 100`  
  输出 `Figs/transition_matrices_{1,2,3}.png`（标题 MLP (RMSE) / MLP (Cross-Entropy)，大字号 \(x_t, x_{t+1}\)）

---

## 实验 2：Kelvin–Helmholtz + U-Net

### 物理 / 数值（Trixi）

- 2D 可压缩 Euler，DGSEM \(p=3\)，Hennemann–Gassner shock capturing，AMR。
- 域 \([-1,1]^2\)，周期边界；基准 elixir：`elixir_euler_kelvin_helmholtz_instability_amr.jl`。
- **同一 IC**，ensemble 成员差异来自 **每步相对噪声** `step_rel_eps=3×10⁻⁴`（Markov 随机一步转移）。
- 输出：`save_dt=0.2 s`，`t=0→5 s`（部分 member 可能 DtNaN 提前终止，保留所有存活帧）。
- **200 members**，磁盘约 **19 GB**（NPZ ~15 GB + H5）。

### U-Net

- 输入/输出：\((4,512,512)\) → \((4,512,512)\)，变量 \((\rho, v_1, v_2, p)\)。
- U-Net `base_ch=32`，**~1.93M 参数**。
- Loss：全局 per-pixel RMSE（归一化后）。
- **划分**：170 train / 10 val（early stopping）/ 20 test（最终 eval）。
- 训练：随机采样 `(member, t)→(t+1)`；**不要**打包单一 `.pt`（16 GB 内存机器易 OOM）。
- 推荐：`batch-size 2 --num-workers 0`，30 epoch，约 4–10 h（RTX 4060 Laptop）。

### 测试比较（20 test members）

三条线：**truth member** | **U-Net rollout** | **ensemble mean**（200 member 同时刻平均 ≈ 条件期望 MC 估计）。

详见 README「Extreme-event comparison」和 `eval_unet.py`。

---

## 计算环境（论文可写）

| 项目 | 配置 |
|------|------|
| Python | 3.13.11 |
| PyTorch | 2.12.1+cu130 |
| CUDA | 13.0 |
| CPU | Intel i7-13620H |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| RAM | 16 GB |
| Julia | 1.11.9（`tools/julia-1.11.9/`，本地，未上 Git） |

---

## Data Availability（要点）

- 代码：https://github.com/DYHAI/GRL_possibility  
- KH：`experiments/kelvin_helmholtz/trixi/`  
- Trixi 引用：Ranocha et al.  
- 200-member NPZ **太大未上传 Git**；需自行跑 `run_ensemble_200.sh` 或从本机拷贝 `outputs/`

---

## 写作 / 待办

- [x] Section 2 理论 LaTeX 草稿（`paper/latex/`）
- [x] Markov methods + 三机制 Discussion 草稿
- [x] KH methods + U-Net 描述草稿
- [x] Jensen 展开 + 非闭合命题
- [ ] 填入 Markov 实验具体 RMSE/KL 数值
- [ ] 完成 KH 200-member 数据 + U-Net 训练
- [ ] 填入 KH high-k ratio、peak ratio、exceedance 数值
- [ ] 主 LaTeX 稿合并进 AGU 模板（用户本地）

---

## 关键引用思路

- Blau & Michaeli (2018) Perception–Distortion：MMSE = 后验/条件均值
- Pangu-Weather, FuXi, GraphCast：MSE 训练范式
- Ranocha et al. Trixi.jl
