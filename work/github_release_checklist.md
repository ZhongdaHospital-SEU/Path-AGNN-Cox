# GitHub 发布清单（Path-AGNN-Cox）

核查时间：2026-08-18。`.gitignore` 已更新，本清单列出提交范围与发布步骤。

## 1. 将要提交的内容（白名单）
- 核心包：`path_agnn_cox/`（models、pathway、loss、train、data、evaluate）
- 复现脚本：`benchmark/`、`work/` 顶层 `*.py`/`*.md`（置换、stdgat 负对照、IMvigor210、外部重连）
- 配置：`config/`、`requirements.txt`、`setup.py`、`README.md`、`.gitignore`
- 教程：`examples/`（quickstart.py + notebook）
- 测试：`tests/`
- 论文与图：`manuscript/`（手稿 md/docx、模板、渲染与核查脚本）、`results/figures/*.svg` + `figure_manifest.json`、`results/immune/`、`results/rewiring_external/`
- 复现主表：`results/*.csv` 中 8 个关键表（benchmark、internal/external C-index、seed、summary）

## 2. 自动排除（勿提交）
- `data/`（~16 GB 原始数据）、`work/pkg/`、`work/models/`、`work/logs/`（共 ~3.4 GB）
- `results/rewiring/{BRCA,LUAD,IMvigor210}/`（置换中间产物 ~2.3 GB）
- 根目录草稿（`Path-AGNN-Cox_*.md`、`PathMoG*`、`题目.docx`、`df.rds`、`results.rds`、`MANIFEST.txt`、`*.tar.gz`）

## 3. 发布前必须替换的占位符（3 处）
1. `examples/Path-AGNN-Cox_quickstart.ipynb` L35：`https://github.com/YOUR-ORG/Path-AGNN-Cox.git`
2. 同文件 L149：`GitHub (placeholder)` 与 `PyPI: path-agnn-cox (placeholder)`
3. `manuscript/cover_letter.md` L25：`[Author names and affiliations]`
4. 手稿 title page：`TBD (author list to be completed by corresponding author)`

替换后重跑：`py manuscript/check_formatting.py`（应 ALL CHECKS PASSED），再 `py manuscript/md2docx.py` 重新生成 docx。

## 4. 发布步骤（需用户账号，CLI 无法代办）
1. 安装 Git for Windows 与 GitHub CLI：`winget install Git.Git`、`winget install GitHub.cli`
2. `gh auth login`（浏览器授权）
3. `git init -b main` → `git add -A` → 检查 `git status` 无大文件 → `git commit`
4. `gh repo create Path-AGNN-Cox --public --source . --push`（建议先建 private 检查，再转 public）
5. PyPI：`python setup.py sdist bdist_wheel` → `twine upload dist/*`（需 PyPI token）
6. 发布后回填正文 Availability 段：把 "the repository URL will be registered upon acceptance" 改为实际 URL，重渲染手稿
7. README 顶部可加 GitHub 徽章（DOI/PyPI 发布后）

## 5. 引用核实（已完成）
- PathMoG：arXiv:2604.24371 真实存在（2026-04-27，Di Wang 等，含 GitHub 源码 https://github.com/wangzoyou/pathmog），正文引用与参考文献条目均已核对，可放心引用。
- 手稿中 "12/25 external cohorts vs 25 for the best baseline" 等数字与 `results/benchmark_results.csv` 一致（正文已如实报告性能不占优）。

## 6. 投稿前作者侧动作
- 填作者名/单位/通讯邮箱（`title_page.docx`、`cover_letter.docx`）
- 建议重启电脑以完全激活 D 盘 24 GB 页文件（重启前确认无训练任务）