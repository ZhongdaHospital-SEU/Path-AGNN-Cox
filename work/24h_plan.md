# Path-AGNN-Cox 论文收尾 24 小时计划（更新于 2026-08-18 04:18）

## 全部完成项
- [x] stdgat 负对照（BRCA 11/54、LUAD 1/54 vs 43/53、3/53）
- [x] 置换检验 1000 次（BRCA P<0.001、LUAD P=0.014）
- [x] 敏感性分析 3 定义稳健（BRCA Ki-67、IMV 风险）
- [x] 3-seed 基准：LUAD 0.50+-0.04、BRCA 0.59+-0.04（正文 3.1 + Limitations 已更新）
- [x] 方法学 2.9 节 + Algorithm 1 + 复杂度 + 环境版本
- [x] 语言平实化 8 处、批注清除（P1-P5）、MKI67 caveat、95% CI
- [x] 图/表 Figure 1-7、Table 1-6、SVG、编号自动；check ALL PASSED
- [x] 交付物：manuscript DOCX（7 图）+ cover letter/title page/highlights/数据可用性 4 个 Word + Colab notebook（占位链接）
- [x] 内存加固：页文件 C 4-8GB + D 24GB、WER 转储、禁更新重启

## 投稿前必办（用户动作）
1. 发布 GitHub 仓库（真实 URL 替换占位符；README 已就绪）+ PyPI 发布 `path_agnn_cox`
2. 核实 PathMoG arXiv:2604.24371 是否真实存在（用户提供，需验证）
3. 填作者姓名/单位/通讯邮箱（title_page.docx、cover_letter.docx 占位符）
4. 建议重启一次激活 D 盘 24GB 页文件（重启前确认无任务）

## 可选补强（暂缓）
- 肿瘤纯度独立锚点（约 1 天，收益不确定）——已用 TMB 独立锚点 + caveat 替代
- Colab 链接：等仓库发布后替换 notebook 内 YOUR-ORG 占位符

## 硬性约束
- 结果如实：不 claim 性能优势；外部队列 0/6、IMV 响应不显著均如实
- 内存 15.4GB：torch 重任务一次 1 个
