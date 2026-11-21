# V15：条件化跨说话人证据与保守纠错——TRAIN-only 可行性诊断

日期：2026-11-21；唯一工程：`/code/CSS`。

## 研究问题与已经得到的负结果

官方 CSS 完整训练作者协议的 IEMOCAP TEST W-F1 已成功复现为
**75.28%**，论文报告 **75.66%**，但原作者代码按每轮 TEST W-F1
选择最优 epoch；这是一种 TEST 参与模型选择的历史基准，
不是严格独立的泛化结果。

此前 V12：CSS 75.28，Self 74.38，Reaction 74.06。
V13：Self 74.75，Cross 74.83，仍低于 CSS。
V14：在 3751 条 A-B-A 的官方 TRAIN 会话内部四折，
A-only 59.02%，Real B 57.71%，Shuffled B 平均 58.18%；
Real-B 模型的 B 被替换后 55.14%。V14 预先设定的筛选失败。
V14 使用固定压缩表示与线性逻辑回归，无法显式学习
“在某类 A 状态下某类 B 信息才有用”的非线性条件交互。

**V15 是一个新的、事前固定的条件交互假设**，
不是在已经看到的 V14 或 IEMOCAP TEST 指标上扫描超参数。
它检查：保留原先 A-only 判断，学习有限幅度的、由 A×B 决定的
预测修正，真实 B 是否能在不同训练会话上稳定优于同容量对照。

## 任务数据：完全不使用官方 TEST

- 只实例化 `IEMOCAPDataset(train=True)`，遍历 Ses01–Ses04
  的 120 段 TRAIN 对话，5810 条话语。
- 复用 V14 已实现且核对过的抽取逻辑，得到 3751 条
  A(previous)–B(intervening)–A(current)，其中情绪发生变化 1166 条。
- **只在 TRAIN 内留一 Session**：依次留 Ses01、Ses02、Ses03、Ses04。
  未留出的三个 Session 用于学习；留出的 Session 只作诊断评估。
- 每段话的输入来自原 CSS SDT 文本/声学/视觉特征，使用
  V14 已固定的无标签随机投影（每个 utterance 224 维）；
  A 部分为 `[A_current, A_previous, A_current-A_previous]`，
  B 部分为 `[B, A_current-B, A_previous-B]`。
- `StandardScaler` 只在每折的训练子集拟合；
  留出会话绝不参与 scaler 或参数拟合。
- `A_previous` 的**真实情绪标签不作为模型输入**。
  训练分类目标为 `A_current` 的真实标签；
  “情绪是否变化”仅用于最终分组分析。
- 打乱的 B 必须来自**同一训练/评估分区**、同一 Session、
  相同 B 说话人角色（M/F）、不同对话；固定打乱种子
  31、47、73，在训练与留出会话分别抽样，不跨折借样。
- 底层原始 pickle 包含 TEST 特征数组，但此程序不迭代
  TEST keys，不建立 TEST DataLoader，不推断或计算 TEST 指标。

## 模型：和 V11/V12/V13 本质不同的改变

1. 先用当前 A、上一个 A 和差分，训练一个小型 A-only
   神经分类器（`672→96→6`，GELU），然后**冻结参数和梯度**。
   所有后续组使用该次训练完全相同的 A-only 模型。
2. 建立一个独立小型 `ConditionalCorrection`：
   A 输入编码成 64 维，B 输入编码成 64 维，
   将 `[h_A,h_B,h_A ⊙ h_B]` 送入纠错门和类别修正头。
   **逐元素相乘**提供 V14 线性分类器没有的条件交互表达能力。
3. 修正必须保守：以 A-only 原 logits 为基础，
   `delta = 1.75 * (1 - max_prob_A) * sigmoid(gate) * tanh(correction)`。
   修正的每一维大小被置信度上界限制；
   越有把握的 A-only 预测，最多允许的修改越小。
   使用系数 0.02 的 `mean(delta²)` 稳定约束。
   初始修正头为全零，因此初始预测与冻结 A-only 完全一致。
   不能使用当前/过去的真实情绪标签或已知的预测对错决定 gate。
4. 所有纠错组**参数量完全相同、随机初始化相同**，
   训练循环、批次顺序和优化器配置一致，仅 B 输入变化：
   - A-only：未增加任何纠错头的冻结参考；
   - Self：纠错头的 B 输入为零（容量控制组）；
   - Real B：纠错头接真实中间 B；
   - Shuffled B：相同纠错头接其他对话的 B；
   - Real B replaced：不重新训练，真实 B 模型评估时将 B 换成打乱 B。
   Self 有相同网络容量但缺失 B；所以 Real 必须优于 Self，
   不能只拿它比冻结 A-only。
5. 模型是**训练集内部的轻量 probe，不是 CSS 主干网络**。
   不会修改 `train_IEMOCAP.py`、`model.py`、PGM、SPF、
   V12/V13 checkpoint，也没有正式 TEST 评测。

## 固定训练设置及预先锁定的决策

- 固定随机种子 61078、61079、61080；
  每一随机种子都完成四个原始 TRAIN Session 留一评估；
  所有预测与折标签逐一对齐。
- A-only 固定训练 24 epochs；
  三个纠错臂分别训练 12 epochs；
  不按留出会话 W-F1 选 epoch，也不调这些超参数。
- Batch size=128；Adam lr=0.001、weight_decay=0.0001；
  隐藏维数 A-only=96、交互分支=64；
  最大修正=1.75、正则系数=0.02。
- 三个训练种子分别汇总一次完整四折 out-of-fold
  W-F1；**不能将 3×3751 个预测作为独立样本夸大样本量**。
- 提前约定：Real B 对冻结 A、Self、Shuffled、
  Real B 被替换这四个比较，三种子平均 W-F1
  都至少高 **0.50 个百分点**；真实 B 相比冻结 A、
  Self 和 Shuffled 在**每个训练种子**的汇总结果都为正；
  且至少 2/3 个训练种子各自在 3/4 留出会话上
  同时胜过冻结 A、Self 和 Shuffled。
- 即使筛选通过，也**不等于证明跨说话人情绪因果影响或顶会创新**；
  要先做 2026 年相关论文创新性排重，明确新增机制，
  然后才在所有 Ses01–Ses04 上进行完整 CSS 训练；
  最终还需要未参与设计的数据或预先锁定的独立评估。
- 如果筛选不通过，**停止本候选**，不通过修改 epoch、
  correction cap、正则权重或种子去追已看过的分数。
  也不能用一次受压缩的轻量实验断言 B 完全没有情绪信息。

## 服务器运行命令

```bash
cd /code/CSS
conda activate cmdpad

git fetch origin
git switch experiment/v15-conditional-cross-speaker-correction

# 单元测试：只用合成数据，不读取 IEMOCAP TEST
python -m unittest discover -s tests \
  -p 'test_v15_conditional_probe.py' -v

# 先验证只有原始 TRAIN 的 3751/1166 A-B-A 结构
python -u -m scripts.run_v15_conditional_probe --preflight

# 一折、一种子、各 2 epoch 的功能测试
python -u -m scripts.run_v15_conditional_probe --smoke

# 全量四折 × 三种子；仍然只使用官方 TRAIN
python -u -m scripts.run_v15_conditional_probe
```

输出：
- `results/v15_conditional_cross_speaker/smoke.json`
- `results/v15_conditional_cross_speaker/report.json`

报告按训练随机种子保存四折和汇总指标、三种对照、
真实 B 被替换的敏感性、各组纠错造成的原预测变化比例，
并输出预先固定的筛选判定。

**研究边界**：V15 不会生成新的 IEMOCAP TEST W-F1。
V14/V15 的 probe 指标不能拿来与 CSS 的 75.28%
或论文的 75.66% 直接比较。
之前 V11–V13 已反复查看官方 Session5 TEST，
后续 IEMOCAP 的 TEST 也不能宣称从未接触。


## 2026-11-21 用户服务器 V15 完整结果：正式 NO-GO

已按照事先固定的 4 个原始 TRAIN Session 留一会话、
3 个训练随机种子 (61078/61079/61080)、冻结 A-only、
12 epoch 匹配容量修正头及同 Session/说话人角色/异对话
Shuffled B 对照完成实验；没有对官方 Ses05 TEST 推断或打分。

所有数值为 TRAIN-only out-of-fold Weighted F1 **百分比**，
不是 CSS 官方 TEST 论文指标：

| Seed | A-only | Self | Real B | Shuffled B | Real B replaced |
|---|---:|---:|---:|---:|---:|
| 61078 | 61.62 | 61.44 | 61.31 | 61.94 | 61.47 |
| 61079 | 63.41 | 63.14 | 63.43 | 63.48 | 62.93 |
| 61080 | 62.27 | 61.65 | 62.20 | 61.60 | 61.32 |

原始运行脚本的无四舍五入汇总差异（Real B 相比）：
- 冻结 A-only：**-0.1139521 pp**；
- Self 同容量修正头：**+0.2387594 pp**；
- 训练时使用 Shuffled B 的同容量修正头：**-0.0225293 pp**；
- Real B 模型、评估时 B 被替换：**+0.4128338 pp**。

三个 seed 的四折中，满足“Real B 同时优于冻结 A、
Self、Shuffled B”的会话数，均未达到 3/4；
`seeds_with_at_least_3_of_4_positive_sessions=0`。
预先固定的四项平均差异门槛均至少 +0.50pp，
实际至少三项没有达到，且 Real B 平均略低于 A-only
与 Shuffled B。

**决策：V15 SCREEN FAILED / NO-GO。**
不应再通过更改 seed、learning rate、训练轮次、
修正强度 1.75、正则系数 0.02 等追逐
同一批开发会话已查看的分数，也不应立即将这条
方案投入完整 CSS 200 epoch 训练。

### 这能证明什么、不能证明什么

- 结果可以支持：在固定的随机压缩特征、A-only 小网络
  和保守非线性纠错头方案中，真实 B 没有带来稳定的
  额外分类收益；即使模型对 B 替换有一定敏感性，
  这种依赖也不等于性能价值。
- 不能证明“B 永远不影响 A 的情绪”；该任务仍限于
  一个受压缩的轻量诊断，且原始 CSS 已经利用双向的
  全对话上下文，因此新增独立 B 分支也可能
  与现有 CSS 的能力重叠。
- V15 的任何数字都不能与 CSS TEST W-F1 75.28%
  或论文 75.66% 直接比较。
- 不要把上述四折/三种子结果当作 12 个独立数据集；
  四折交叉使用同一原始 TRAIN 语料，随机种子只是
  重复初始化。

下一步仅建议 **无训练的误差审计**：
从已保存 `results/v15_conditional_cross_speaker/report.json`
的 `seed_results[*].pooled_labels`、
`pooled_predictions` 与 `pooled_changed_for_posthoc_only`
核对 Real B vs A-only/Self/Shuffled 的修正数、
新增错误数、对话情绪变化组的表现及
`folds[*].prediction_flip_fraction`。
不能利用这些分析再调节同一个方案的阈值。

长期研究应先重新审视任务定义和独立的新证据，
在文献排重的基础上提出**新的可证伪机制**，
而不是把 V11–V15 失败方案微调为“V16”。
原 IEMOCAP Ses05 TEST 已被多次使用，未来论文中的
正式性能证明需要披露历史并最好增加独立数据集评估。

用户服务器上的原始结果位置（此提交没有重新计算）：
`results/v15_conditional_cross_speaker/report.json`。


### 只读误差审计（无需新训练、无需 TEST）

为了区别“纠错头几乎没改变输出”与“纠错头改了却
新增错误更多”，新增 V15 只读报告审计脚本：

```bash
cd /code/CSS
conda activate cmdpad
git pull --ff-only origin experiment/v15-conditional-cross-speaker-correction
python -m unittest discover -s tests \
  -p 'test_v15_train_only_audit.py' -v
python -u -m scripts.audit_v15_train_only
```

输入仅使用已有的
`results/v15_conditional_cross_speaker/report.json`，
从三种子逐条 out-of-fold TRAIN 预测计算
Real B 相比 A-only / Self / Shuffled B / B 替换模型
的 fixed、broken、net_more_correct、
raw_prediction_flips；并分别报告 changed/stable
（真实变化标签仅作分析）。**无模型加载、无训练，
不访问官方 TEST。** 结果输出到
`results/v15_conditional_cross_speaker/audit_posthoc.json`。
该审计只能解释已锁定 V15 的失败机制，不可以反过来
用于阈值、随机种子、epoch 或纠错强度选模。
