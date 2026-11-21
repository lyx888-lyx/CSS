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
