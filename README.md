# CGEC-Faith：中文大模型纠错失败模式数据集 + 忠实度判别器

> **CGEC-Faith** 是首个标注「大模型中文语法纠错**失败模式**」的数据集，并训练一个「纠错忠实度判别器」，证明 F0.5 这类传统指标量不出的失败模式是**可标注、可学习、可诊断、可作质量闸门**的。

## 概览

- **数据集**：2000 句中文病句（NLPCC-2018 测试集）× deepseek / qwen / glm 三个大模型各纠错一次 = **6000 条** `(原句 → 改句)`，人工标注 5 类失败模式。
- **判别器**：基于 `hfl/chinese-roberta-wwm-ext` 训练，二分类（忠实 / 不忠实）+ 五分类（失败类型）。

## 5 类失败模式

| 标签 | 含义 |
|---|---|
| 忠实正确 | 改后句子正确、通顺、意思没变 |
| 过度纠正 | 把对的改错 / 改得更不通顺 |
| 语义漂移 | 改动改变了原句意思 |
| 幻觉 | 凭空添加原句没有的内容 |
| 漏改 | 该改的错没改，句子仍有错 |

## 主要结果

- **标注一致性**：双标子集 κ = 0.7539；全量定稿 **κ = 0.9682**（一致率 98.53%）。
- **判别器**（test 600 条，模型未见数据）：
  - 二分类：Accuracy 0.7417；忠实 F1 0.8264 / 不忠实 F1 0.4951（vs「全判忠实」基线 0）。
  - 五分类：Accuracy 0.6667；忠实正确 F1 0.7787 / 漏改 0.4292 / 语义漂移 0.2418 / 过度纠正 0.50 / 幻觉 0。
- **区分度**：F0.5 几乎相同的模型（deepseek 0.328 / glm 0.327，Δ=0.0008），失败模式分布 **TVD = 0.21**。
- **质量闸门**：判别器做后处理过滤，把大模型输出忠实率 **68.67% → 76.72%（+8.05pp）**，合格句保留率 89.56%。

## 数据集

- 文件：`results/full/标注表_全量_定稿.csv`（6000 条，6 列：`序号 / 原句 / 模型 / 大模型改句 / 你的标注 / 复核判定`）。
- 双标子集（300 条，用于算 κ）：`results/annotation/双标_300_我.csv`、`双标_300_同学.csv`。
- 编码：**GBK**（Windows 简体中文）。读取用 `pandas.read_csv(..., encoding="gbk")`，或编辑器选 GBK 打开。
- 来源：原句取自 **NLPCC-2018 中文语法纠错评测测试集**（引用见下）；改句由 deepseek/qwen/glm 生成；标签为人工标注。
- License：**CC BY 4.0**（见 `DATA_LICENSE.md`）。

> **NLPCC-2018 引用**：Yuanyuan Zhao, Nan Jiang, Weiwei Sun, Xiaojun Wan. *Overview of the NLPCC 2018 Shared Task: Grammatical Error Correction*. NLPCC 2018.
> 原始数据下载：http://tcci.ccf.org.cn/conference/2018/ （测试集 `source.txt` + 参考答案 `gold.0.txt`）。

## 复现说明（哪些需要额外下载 NLPCC 数据）

- **训练判别器 + 质量闸门**：只需本仓库的定稿 CSV + 双标数据，无需额外下载。
- **计算 κ**：`python scripts/compute_kappa.py results/annotation/双标_300_我.csv results/annotation/双标_300_同学.csv`。
- **区分度 / 数据收集**：需要 NLPCC-2018 测试集，从上面链接下载后放到 `data/raw/nlpcc2018/extracted/Data/test/`（含 `source.txt` 与 `gold/gold.0.txt`）。

## 安装

```bash
git clone <仓库地址>
cd zh-gec-llm
pip install -r requirements.txt   # 仅判别器训练需要 torch/transformers/accelerate/scikit-learn/numpy；数据与评分只用标准库
```

## 快速上手

### 1. 配置 API（复现「数据收集」才需要）

```bash
cp config.example.py config.py   # 填入你的 deepseek/qwen/glm api_key
```

### 2. 切分数据（训判别器）

```bash
python scripts/prepare_train_data.py   # 定稿 CSV → results/discriminator/{train,val,test}.jsonl + label_map.json
```

### 3. 训练判别器（需 GPU，一张 T4 即可）

```bash
python src/train_discriminator.py --task bin     # 二分类：忠实 / 不忠实
python src/train_discriminator.py --task multi   # 五分类：失败类型
```

### 4. 质量闸门评估

```bash
python scripts/eval_quality_gate.py   # 用 test_predictions.csv 算过滤前后的忠实率
```

其它：`compute_kappa.py`（算标注一致性 κ）、`verify_discrimination.py`（区分度验证）、`run_experiment.py`（单次纠错 + F0.5 打分）。

## 目录结构

```
zh-gec-llm/
├── README.md / LICENSE / DATA_LICENSE.md
├── 方案.md / 日志.md / 证据链与实验方案.md / 训练判别器.md / 标注指南_v3.md
├── requirements.txt
├── config.example.py          # API 模板（真实 key 放 config.py，已 gitignore）
├── data/                      # NLPCC-2018 原始数据
├── src/
│   ├── client.py              # 调 LLM API（纯标准库）
│   ├── prompts.py             # 5 种提示策略
│   ├── collect.py             # 批量收集大模型改句
│   ├── run_experiment.py      # 单次实验 + F0.5 打分
│   ├── train_discriminator.py # 训忠实度判别器
│   └── eval/m2_scorer.py      # 字符级 MaxMatch F0.5 评分器
├── scripts/
│   ├── prepare_train_data.py  # 数据切分（核心）
│   ├── compute_kappa.py       # 双标 κ（核心）
│   ├── verify_discrimination.py # 区分度（核心）
│   ├── eval_quality_gate.py   # 质量闸门（核心）
│   └── ...                    # 其余为一次性分析脚本
├── results/
│   ├── full/标注表_全量_定稿.csv  # ← 数据集（6000 条）
│   ├── discriminator/         # 训练切分 + 训练产出 + 报告
│   └── ...
└── references/                # 相关论文
```

## 引用

论文尚未发表。引用本项目时请标注：CGEC-Faith（2026），作者 念依恒。正式 BibTeX 待论文发表后补充。

## License

- **代码**：MIT License（见 `LICENSE`）。
- **数据集**：CC BY 4.0（见 `DATA_LICENSE.md`）。
