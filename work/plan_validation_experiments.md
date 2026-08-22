# ② 强外部生物学验证方案（GDSC 真实药敏 + ICB 免疫治疗队列）

目标：为“通路重连”提供独立于 in-silico 预测与训练数据的生物学证据，支撑冲 Briefings/CSBJ 的审稿论证。
验收标准：至少一个方向的验证显著（FDR<0.05），或 ≥4 个 ICB 队列方向一致。

---

## 方案 A：GDSC2 真实药敏验证（推荐先做，数据已就绪，约 2–3 天）

### 科学问题
“重连信号强的通路基因特征”是否与真实实验 IC50（GDSC2，而非 oncoPredict 预测）相关？
若成立，直接把 3.5 的 in-silico 预测（Table 9）升级为“预测方向与真实药敏一致”。

### 数据（全部在本地，无需下载）
- 表达：`work/pkg/GDSC2/DataFiles/DataFiles/Training Data/GDSC2_Expr (RMA Normalized and Log Transformed).txt`（约 1000 细胞系 × 12000 基因）
- 药敏：`work/pkg/GDSC2/DataFiles/DataFiles/GLDS/GDSCv2/complete_matrix_output GDSCv2.txt`（细胞系 × 药物 IC50）
- 注释：`Cell_Lines_Details.xlsx`（组织/亚型）、`screened_compunds_rel_8.2.csv`（药物名/靶点）
- 重连通路基因集：`results/rewiring/BRCA/pathway_test.csv`（43 个显著通路）、`results/rewiring/KIRC/pathway_test.csv`（52 个）
- 通路-基因映射：`data/processed/rewiring/`（模型同款 primary-pathway 分配）

### 分析流程
1. 取 BRCA/KIRC 显著重连通路（BH-FDR q<0.05）的基因集，用模型同款 primary-pathway 分配避免基因多通路重复。
2. 细胞系“重连评分”近似：GNN 权重不能直接迁移，采用保守的通路层近似——每个细胞系计算通路内基因两两共表达的加权和（权重取 TCGA 训练模型的通路内平均注意力，作为拓扑先验），标准化为 z 分数；另加两个对照评分：等大小随机基因集、静态通路（无注意力权重）。
3. 关联检验：每个（通路评分 × 药物 IC50）对做 Spearman，BH-FDR 校正；优先看 3.5 中已报告的 17 个 curated 化合物（与 Table 9 直接可比）。
4. 分层复核：按细胞系组织类型（乳腺癌系、肾癌系）重复，检验效应是否组织特异。
5. 方向一致性：真实 IC50 关联方向与 Table 9 in-silico 方向一致者标记为“预测-实验一致”。

### 产出与入稿方式
- 新增 1 个 Figure（GDSC 验证：Top 通路 × 药物关联热图 + 方向一致性散点），插在 3.5 后。
- 正文 3.5 扩写一段（约 1 段）；若显著，Discussion 升级为“重连评分的药理学可解释性”。
- 不新增表：结果以 Figure + 正文数字呈现（保持 Table 1–9 不变）。
- 阴性预案：若 FDR 均不显著，写为“真实药敏关联方向与预测一致但未达 FDR 阈值”，仍保留为探索性证据，不改论文定位。

### 风险
- 细胞系与肿瘤组织状态差异大，效应量可能小；方案已用“保守近似 + 多重对照”兜底。
- 通路基因集之间重叠；primary-pathway 分配与模型一致，可辩护。

---

## 方案 B：ICB 免疫治疗队列响应 meta 分析（需下载，约 3–5 天）

### 科学问题
重连幅度是否与免疫检查点阻断响应相关？IMvigor210 单独 P=0.111 不显著，多队列 meta 可提供合并证据。

### 数据
- 本地已有：IMvigor210（`data/processed/IMvigor210/`，n≈348，响应+OS+Ki-67）。
- 需下载（复用 `work/download_geo_parallel.py`）：
  - GSE91061（Riaz 2017，黑色素瘤 anti-PD-1，n≈42）
  - GSE78220（Hugo 2016，黑色素瘤 anti-PD-1，n≈28）
  - GSE100797（Lauss 2017，黑色素瘤，n≈28）
  - GSE135222 或原数据（Gide 2019，n≈91，若 GEO 不可用则从文章附件）
  - Liu 2019（Cell，n≈121，需从 Cell 补充材料获取表达+响应表）
  - 可选：Jung 2019（胃癌，n≈45）
- 全部为公开去标识数据，符合伦理声明。

### 分析流程
1. 每队列用预训练模型（BRCA/LUAD 权重）或简化重连评分计算患者重连幅度；报告为探索性跨癌种迁移。
2. 队列内检验：CR/PR vs SD/PD 的 Wilcoxon；高/低重连 OS 分层（log-rank + Cox）。
3. 跨队列随机效应 meta：效应量为标准化中位差（Hedges g），报告合并效应量、95% CI、I² 与合并 P。
4. 敏感性：去掉任一队列、仅黑色素瘤队列、固定效应模型。
5. 辅助锚定：合并队列内重连幅度与 Ki-67/TMB 的相关（若可得）。

### 产出与入稿方式
- 新增 1 个 Figure：森林图（各队列效应量 + 合并菱形）+ 各队列响应箱线图。
- 正文 3.4.3 的 IMvigor210 句改为“meta 分析合并结果”；Discussion 升级为“重连幅度与 ICB 响应跨队列一致（探索性）”。
- 若合并 P<0.05，这是冲 Briefings 的最强新增证据。

### 风险
- 队列间平台/批次差异与跨癌种迁移：正文明确“探索性、需前瞻验证”。
- 小样本队列（n≈28）效应不稳定：meta 报告 I² 并做 leave-one-out。

---

## 执行顺序与时间线
1. 方案 A（2–3 天）：数据就绪，先跑；产出 GDSC Figure + 3.5 扩写。
2. 方案 B（3–5 天）：下载 4–6 队列 → 预处理 → meta；产出森林图 + 3.4.3 升级。
3. 若 A、B 至少一个显著：升级 Discussion、重渲染、更新 Table 9 讨论、提交并推送。
4. 若均阴性：维持“框架稿”定位，投 CSBJ/BMC 档位，并把阴性写入 Limitations（保持诚实卖点）。

## 里程碑检查
- [ ] A：≥1 个（通路, 药物）在真实 IC50 上 FDR<0.05 且方向与 in-silico 一致
- [ ] B：合并效应量 P<0.05 且 I²<75%，或 ≥4 队列方向一致
- [ ] 产出并入正文后：check_formatting.py 全过、docx 重建、提交推送
