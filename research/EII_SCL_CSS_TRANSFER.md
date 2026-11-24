# CSS × EII-SCL 迁移实验（IEMOCAP TRAIN-only）

## 研究背景

EII-SCL: Harnessing Emotional Inertia for Multimodal Emotion Recognition in
Conversation, arXiv:2607.17366v1 (2026-07-19)，用户上传 PDF 第2–3页
公式(5)–(12)。使用同一说话人、真实情绪、惯性窗口和动态权重构造正负样本。
本文仅迁移训练目标到 CSS，不复现论文 MM-Transformer/MM-DialogueGCN 主干、
特征编码器或论文全五 Session LOSO 指标。

先前固定四折原始 CSS 的 OOF W-F1=69.251%；历史遮蔽显示近期自身历史对
情绪维持有益、对部分转折有干扰；HIUL 推理选择器整体 W-F1 -0.294pp，
按预定标准 NO-GO。本实验不继续优化 HIUL，也不先加入自研干预效用权重。

## 冻结四组对照

A. baseline：直接复用已有 seed61078 原始 CSS 四折固定200 epoch报告和
checkpoint；训练/留出会话划分、标签顺序、SHA256 必须吻合。不重新训练A。

B. supcon：同一对话内，所有同情绪样本为正样本（不限制说话人）；
不同情绪样本为负样本，标准逐正样本平均 log-softmax SupCon。
C. eii_fixed：论文式正负集合与动态负样本权重，固定惯性窗口=2（全对话
轮次索引距离；参考已观察的近期自身历史平均 lag 约1.99）。
D. eii_dynamic：相同正负集合与权重，使用注意力加权距离再向下取整的
动态惯性窗口；**固定单位 Q/K 投影是本实验的明确近似，而非官方实现**。

### 未说明的复现细节及处理

- 按论文(5)、(9)，同说话人同标签为正样本（不要求在窗口内），其他说话人
  即使同情绪也按 easy-negative 处理；同说话人不同情绪且位于窗口内为 hard-negative。
- 所有样本对严格在同一个对话内，排除自己和 padding，不把不同对话的 M/F
  当成相同真实说话人。窗口为对称的绝对轮次距离，与论文一致，非因果预测。
- 论文(6)称 Wq/Wk 可学习，(7) floor 与后续离散集合无常规梯度路径，
  文中未说明估计器或额外监督。D 采用 Wq=Wk=I 的原始未缩放点积，
  no_grad 计算窗口，无随机或伪可学习投影；记录平均窗口和 hard pair 数。
  不将它称为官方动态窗口的严格复现，也不宣称解决了可微性。
- 原文(10) hard 权重 (1-cos)/2 直接参与反传，不 detach，按公式实现；
  高相似度区间可能出现梯度方向变化，记录为复现假设而非论文已验证缺陷。
- 无正样本的锚点跳过，有正样本锚点在 batch 中平均；整个 batch 无正样本
  时返回图连接的零损失。原文对空集合及聚合方式没有交代。
- B/C/D 共用 tau=.07、alpha=.02；CSS 原始 PGM 三任务权重计算不变，
  在 PGM 总损失之后追加 alpha*L_aux，因而辅助损失仍影响总梯度。
- CSS 分类器前融合特征通过训练期 pre-hook 获取，不修改模型结构、
  9项 forward 返回值、推理路径或已有 checkpoint 参数键。
- 全部使用原始 CSS 特征、seed61078、batch32、200 epoch、Adam lr1e-4、
  wd1e-5、原始类别权重及固定最终 raw 权重，不做 SWA/EMA。
  原论文使用最后10个checkpoint平均；为公平复用A，本迁移实验不使用该协议。
- 只使用 Session1–4 训练划分的 120 对话 / 5810 话语。每折训练三个
  Session，固定第200轮后仅一次留出评估；不实例化官方 Session5 TEST dataset，
  不以留出分数选 epoch。所有会话已有开发暴露，仍是探索性结果。
- 本轮只验证 IEMOCAP，不训练 MELD，不加入 IGEI-CL/ISSD。

## 服务器运行

cd /code/CSS
conda activate cmdpad
git status --short
git fetch origin
git switch experiment/eii-scl-css-transfer

python -m unittest discover -s tests -p 'test_eii_scl_transfer.py' -v
python -u -m scripts.run_eii_scl_transfer --preflight
python -u -m scripts.run_eii_scl_transfer --smoke

smoke 对每个 B/C/D 使用两个训练侧对话，真实 CSS 前向、PGM 和反向传播，
不保存权重，不评分留出集。预先指定 Ses01 为首轮 pilot：

for arm in supcon eii_fixed eii_dynamic; do
  python -u -m scripts.run_eii_scl_transfer --fold Ses01 --arm "$arm" || break
done
python -u -m scripts.run_eii_scl_transfer --pilot-summary

只在所有三组完整结果都存在时汇总。先看相同预算下 EII 对原始 CSS 及普通
SupCon 的整体 W-F1 和转折/维持取舍。若两个 EII 均不超过 CSS，则不在
Ses01 上重新选超参数，先评估停止该迁移路线；若出现有意义信号，再决定
继续其余三折，不把单折当作稳定结论。不要只保留表现好的组。

其余三折如继续，均使用完全相同的固定参数：

for s in Ses02 Ses03 Ses04; do
  for arm in supcon eii_fixed eii_dynamic; do
    python -u -m scripts.run_eii_scl_transfer --fold "$s" --arm "$arm" || exit 1
  done
done
python -u -m scripts.run_eii_scl_transfer --aggregate

输出 results/css_eii_scl_transfer/seed61078/SesXX/ARM/{epoch200.pt,report.json}
以及 pilot_summary.json / cross_session_summary.json。
汇总保存整体、changed、stable、first、per-class F1、混淆矩阵、
三组定向类别误判率，以及 fixed/broken 对比。changed/stable 子集使用全部
有同说话人历史的样本（1610/3960），不同于早先遮蔽审计的匹配子集
(1597/3905)。报告标签仅事后评估。不得直接与原论文74.01%或 CSS Session5
75.66%相比较。保存后拒绝覆盖；若训练中途异常，先检查目录与异常原因。

## 验证状态

代码、CPU 合成测试及协议已提交。当前环境无法直接访问用户服务器数据/GPU，
未执行服务器单元测试及真实训练，不包含性能结论。
