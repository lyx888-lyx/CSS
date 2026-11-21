# CSS 新方向：四分类头互补性审计 + SWA/EMA 原始模型权重平均

日期 2026-11-21。仅使用已有 `/code/CSS`，不再叠加 V11–V15 的跨说话人特征/纠错头。

## 为什么先做这两件事

已复现 CSS 原始作者式 IEMOCAP Session5 TEST W-F1 **75.28%**
（论文 75.66%），但作者原始训练代码每一 epoch 查看 TEST
并选择其中最好的一轮；该基准不是独立的严格 TEST 选模。
完整 TRAIN 原始 CSS 在 epoch 200 的 TRAIN W-F1 曾达 91.88%、
TEST 74.04%，历史最佳 TEST 位于 epoch 106；
这种跨会话泛化落差与末轮波动值得诊断，但并不单独证明过拟合。

V11–V15 的额外自/他人声学差分、预测辅助损失、线性与非线性
A-B-A 纠错实验都未显示稳定提升。本实验不尝试继续修改此机制。

本分支同时提供两项**互相独立**的研究工具：
1. 四分类头审计，读取已有 V10 Session4 原始 CSS 验证 checkpoint，
   不训练网络、不调用 TEST；
2. 完整 TRAIN 原始 CSS **同一次训练**追踪 Raw/EMA/SWA，
   无新模型/损失/参数/额外训练分支，训练中不访问 TEST。
   只在实验彻底冻结后允许显式 TEST one-pass 指标对照。

不能把 EMA/SWA 当作 2026 年新的理论创新，它们是
成熟的优化/泛化技术；四头互补性也只是后续
误差机制创新的依据，而不是性能已提升的证据。

## A. 四头互补性审计：完全不重新训练

输入既有用户服务器 checkpoint：
`results/v10_iemocap/paired/original_valid_best.pt`。

严格检查：这是 **V10 original CSS** （不含新增文本特征），
TRAIN Ses01–03 90 段/4298 句；Session4 验证
30 段/1512 句；epoch 依据原有 VALID W-F1
确定。加载已有 checkpoint，不从头训练也不会搜索
其他 epoch。不要拿 V12 在 Ses01–04 全训之后
TEST-best 的模型在训练过的 Ses04 上审计，
否则产生训练集拟合偏差。

使用原 CSS `model.forward` 中四个
`[text, audio, visual, fused]` 的 log_prob，
`outputs[4]` 是 fused 概率；验证四头归一化和
输出形状，计算（按完整 1512 句计算）：
- 每一头 ACC/W-F1/Macro-F1 和六类 F1；
- 融合分类头错、任一单模态头对的潜在纠正上界；
- 融合分类头对、某单模态头错的误修风险；
- 任一头与融合头 disagree 的次数，类别级数量；
- `fused wrong + some unimodal correct` 与
  `fused correct + some unimodal disagree` 的对比。

所有“单模态头能救回来”的结论都是用真实标签
**事后确认**，即 ORACLE 统计，模型在推理时
不能知道哪一头正确；这是候选研究空间的上界而非
可部署的性能增益。如果单模态头很少覆盖
融合头的错误，则暂时不做后续多分类头融合。

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/css-head-audit-weight-averaging

python -m unittest discover -s tests \
  -p 'test_css_four_head_audit.py' -v

python -u -m scripts.audit_css_four_heads
```

若原 V10 checkpoint 路径不同，
允许显式给 `--checkpoint /实际路径/original_valid_best.pt`，
但代码仍强制检查 V10 original 配置和 Ses04 真正留出。
结果：
`results/css_head_audit/session4_original_heads.json`。

## B. SWA/EMA：一个完整 CSS 训练轨迹，三个固定模型

原 CSS 不变：
- IEMOCAP 全 TRAIN Session1–4，120 段 5810 句；
- seed=61078, 200 epochs, batch=32；
- Adam lr1e-4, weight_decay1e-5；
- 原始 CSS Transformer、PGM 三任务梯度平衡、SPF、
  文本/音频/视觉特征及六类权重；
- 不改变 `model.py` 或 `train_IEMOCAP.py`；
- 不使用 TEST 提前停止、调学习率或挑选 epoch。

完全预先确定的三组：
- `raw_final`: 第 200 个 epoch 结束时原 CSS 权重；
- `ema_final`: epoch100 结束时开始，
  每完成一轮更新 EMA，decay=0.90，
  总计 101 次（epoch100–200）；
- `swa_final`: epoch151–200 共 50 个
  epoch 结束权重的简单算术平均。

参数与浮点缓冲区在 CPU shadow 中更新，
非浮点 buffer 复制最近值；原模型网络与
Adam 优化状态完全不变。代码会核对三份
state_dict 的参数名/形状相同。没有 BatchNorm，
无需额外 BN 统计更新。

**公平性重要区别：**
历史 CSS 75.28% 是“每轮 TEST 选最佳”的 W-F1，
本实验固定 epoch 200，并且训练期间
**从不运行 TEST 推理**。因为原作者过去每轮
还遍历了 VALID/TEST dataloader（会消耗随机数），
新训练运行的原始末轮数字也不能保证
与 V12 的 epoch200 74.04 完全一致。
只有**同一次 run 的 raw_final 与
ema_final/swa_final**是严格训练轨迹匹配的对照。

### 训练命令（不查看 TEST）

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/css-head-audit-weight-averaging

python -m unittest discover -s tests \
  -p 'test_css_weight_averaging.py' -v

python -u -m scripts.run_css_weight_averaging --preflight

# 完整 5810 条 TRAIN 一轮，检查原 CSS PGM/backward/EMA/SWA，
# 完全不构造 TEST 数据集：
python -u -m scripts.run_css_weight_averaging --smoke

# 唯一 200 epoch 原始 CSS 全 TRAIN + CPU EMA/SWA：
python -u -m scripts.run_css_weight_averaging --train
```

结果：
```text
results/css_weight_averaging/
  smoke/
    manifest.json
    history.json
  full/
    history.json
    manifest.json
    raw_final.pt
    ema_final.pt
    swa_final.pt
```

以上训练命令**不能评价 TEST**；
代码不调用 `IEMOCAPDataset(train=False)`。

### 明确的一次性 TEST 数字对照（可选）

完成并冻结 200 epoch 的 Raw/EMA/SWA 参数后，
如果决定进行历史作者式 TEST 上的**探索性**
数值比较，再显式运行：

```bash
python -u -m scripts.run_css_weight_averaging \
  --evaluate-test-once --acknowledge-prior-test-exposure
```

这个命令才会加载官方 Session5 TEST 40 段、
1623 句，并且对**三个固定结果全部打分一次**；
一份 `official_test_onepass_report.json` 保存
ACC/W-F1/Macro-F1、每类 F1、逐条预测以及
EMA/SWA 相对原始同训练轨迹末轮模型的
fixed/broken 纠错数量。
再运行时如文件已存在，代码拒绝覆盖。

这不是未接触的独立 TEST：V11–V13 已反复查看过
Session5 的结果，V12 还通过 TEST 选择过 epoch。
请不要从这张表挑最优衰减率、SWA 窗口或训练轮次
再重复评估；要在论文里声称方法可靠，
需要另一个未参与开发的新数据集或预先固定的
独立对照设计。

## 提前写明的判断方式

A 审计：先看历史固定 Session4 上，
融合头错误被其他头单独预测正确的频率，
与融合头原本正确却被其他头反对的频率。
结果是研究空间的**诊断**，不是可以直接实现的
正确性上界提升。

B 泛化基线：`EMA/SWA` 至少要在同一次
200 epoch TRAIN 后超越 `raw_final`，
才能被认为在本次探索性 TEST 数值上
对权重平均有积极信号；不能拿它们直接
对比历史按 TEST 挑的 75.28% 来宣称
公平或独立的新方法改进。

如果两个方向都没有明确收益，则停止，
不为了追分扫描 EMA decay、SWA 起点、
类别权重或未来的 TEST-best epoch。

**两部分都未在用户 GPU 上实际运行。**
单元测试脚本已提交，务必先执行再正式训练。
