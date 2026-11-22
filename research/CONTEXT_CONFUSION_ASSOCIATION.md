# V17 — TRAIN-only 四折 OOF 对话上下文 × 情绪混淆关联审计

## 动机

上一阶段的固定第 200 epoch、IEMOCAP 官方 TRAIN-only 四折留一 Session 实验：
5810 条话语，1785 条错误。三个已根据历史 Session4 审计选定的类别对：
Angry↔Frustrated：435 条；Happy↔Excited：329 条；
Neutral↔Frustrated：356 条；共 1120 / 1785 = 62.75%。

出现次数可重复不等于原因已知。本阶段不训练模型，只检查
这些定向混淆是否与三类**事后可观察**的对话条件有关。

## 诊断定义

每条评估话语在原始 TRAIN 的逐话语对齐 OOF 预测基础上，沿同一对话向前寻找同一说话人的上一次话语。对有历史者，定义：

1. own_emotion_changed：当前真实情绪是否不同于上一次同说话人真实情绪；
2. other_intervened：上一次同说话人话语到当前话语之间，是否出现另一说话人；
3. previous_own_label_matches_wrong_target：上一次同说话人真实情绪是否恰好等于所分析的误判目标类别。

对六个**定向类别** A→B，固定当前真实类别 A，
在每种条件 TRUE/FALSE 内统计“错误预测为 B”的数量、
支持数和错误率；既提供全四折汇总，也逐 Session 提供。
不对没有同说话人历史的样本假设改变状态；空分组保留 null。
注意双向类别对的分母不同，不能只看错误个数。

## 严格约束

- 只加载 IEMOCAPDataset(train=True)，并按原始官方 TRAIN keys 顺序恢复四折预测。不会实例化 TEST dataset 或训练/加载模型。
- 数据需覆盖 Ses01–04 的 120 段对话、5810 句，并与现有 cross_session_summary.json 和每折 OOF 标签逐条吻合；验证四折 checkpoint SHA256 记录与汇总一致，不重新打开 checkpoint。
- 上述三种条件使用**真实历史情绪**只用于统计分析；不向 CSS 推理输入真实历史标签。不将错分目标 B 作为可部署的纠错判断。
- CSS 原版使用双向对话上下文，即使出现统计关联，也**无法证明过去的 B 发言导致 A 情绪变化**，或证明模型只关注了过去。
- 三组类别对先前已通过 Session4 选出，后续分析仍是探索性，不报告独立泛化、p 值或因果效应。
- 只读取原始 JSON，写新 JSON，不改原先的 fold/checkpoint/summary；若已有新报告会拒绝覆盖。

## 使用方法

~~~bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/context-confusion-association

python -m unittest discover -s tests \
  -p 'test_context_confusion_association.py' -v

python -u -m scripts.audit_context_confusion_association
~~~

新输出：

results/css_session_confusion_stability/seed61078/context_confusion_association.json

## GO / NO-GO 的原则

- 如果某个 A→B 定向错分在 4 个 Session 中都与 **情绪变化 / 前一自我情绪等于错分类**呈一致关系，值得先测试简单的上下文基线，而不是立刻做新网络。
- 如果只有 pooled 差异、但各 Session 差异方向相互矛盾，应优先怀疑分布/支持数差异，停止以该相关性构造模型。
- other_intervened 存在关联仍不证明 B 信息有独立预测价值；必须和同容量 A-only、打乱 B、对话其他历史等负对照比。
- 如果三类条件都不稳健，回到当前话语的细粒度多模态判别，不再重复 V11–V15 的条件纠错设定。

**状态**：代码及合成数据单元测试已加入 GitHub 分支；尚未在用户的真实 TRAIN 四折预测环境中执行。
