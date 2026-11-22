# CSS 跨 Session 情绪混淆稳定性审计（TRAIN-only）

## 目的

此前在已经检查过的 IEMOCAP Session 4 上发现三组常见混淆：
Angry ↔ Frustrated 157 条；Happy ↔ Excited 100 条；
Neutral ↔ Frustrated 97 条。**这些候选类别对已依据 Ses04 结果选择，不能视为独立验证假设。**

本实验只验证这些混淆是否在原始 TRAIN 的多个 Session 上反复出现，不改变 CSS、不主张新模型性能提升，也不将相关性解释成说话人情绪影响的因果关系。

## 数据与冻结协议

- 仅按 IEMOCAPDataset(train=True) 的官方 TRAIN keys 遍历 Ses01–04，120 段对话、5810 句；四个 Session 对话数分别是 28、30、32、30。
- 每折从头训练一次原始 CSS：训练三个 Session，留出其余一个 Session；固定 **200 epoch**，只在训练结束后进行一次留出 Session 评估。不使用留出分数选择 epoch。
- 原始 create_model('original')，原始 PGM、SPF、损失、类别权重、Adam lr=1e-4/weight_decay=1e-5、batch32。
- 默认 seed 61078；四折组成 TRAIN-only out-of-fold 结果。单种子不代表训练随机性稳健。
- 每 Session 记录 ACC、W-F1、Macro-F1、混淆矩阵，以及三组候选类别对的**定向错误率**：A→B 数量 / A 真实类别支持数；B→A 同理。并报告双向错误率和四折出现次数。
- 官方 Session5 TEST 不构造 DataLoader、不评估、不参与模型选择。原始 pickle 格式自身包含其他分区数组，但脚本不遍历 TEST keys。

## 运行命令

在 /code/CSS 服务器中：

~~~bash
conda activate cmdpad
git fetch origin
git switch experiment/session-confusion-stability-audit

python -m unittest discover -s tests -p 'test_session_confusion_stability.py' -v
python -u -m scripts.run_session_confusion_stability --preflight
python -u -m scripts.run_session_confusion_stability --smoke

# 四个独立全量训练，每折 200 epoch；完成后才能聚合
python -u -m scripts.run_session_confusion_stability --fold Ses01
python -u -m scripts.run_session_confusion_stability --fold Ses02
python -u -m scripts.run_session_confusion_stability --fold Ses03
python -u -m scripts.run_session_confusion_stability --fold Ses04

python -u -m scripts.run_session_confusion_stability --aggregate
~~~

输出路径：
- results/css_session_confusion_stability/seed61078/Ses01.json 等
- results/css_session_confusion_stability/seed61078/Ses01_epoch200.pt 等
- results/css_session_confusion_stability/seed61078/cross_session_summary.json

四折汇总包括 per_session、pooled_out_of_fold、pair_stability、权重校验和及研究解释限制。再次运行拒绝覆盖已有 fold/summary。

## 如何使用结果

- 若两种定向错误在多个留出 Session 上都频繁出现，可调查说话人情绪变化、语义差异和模态证据是否与这种错分有条件关联。
- 若高度集中于单一 Session，应先调查该 Session 的类别分布和跨会话偏移，而不是直接增加困难类别损失。
- 四折指标**不可与论文官方 TEST W-F1 直接比较**，也不能挑一个 fold 的最优 epoch 或凭一个 seed 宣称稳健。
- 之前对 Session4 和官方 Session5 都有研究暴露，后续独立泛化论证必须披露历史，最好还有新的未参与开发的数据或独立冻结评估协议。

**状态**：纯统计模块已通过合成数据 CPU 单元测试；实际 IEMOCAP 的四折 GPU 训练尚未执行，因此没有真实四折结果。
