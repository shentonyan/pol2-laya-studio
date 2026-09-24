# 蒸馏数据报告

生成时间：2026-09-24T04:48:06+0800

> 诚实条款：gold 是老师模型的共识分布，不是人工真值。学生模型学到的上限是老师，老师的偏差也会一起学走。判断质量必须用独立的人工金标集评估（`python -m distill annotate`）。

## 设置

- 老师：`jev:openjev`（权重 1.0）, `ollama:qwen2.5:7b`（权重 1.0）, `ollama:llama3.1:8b`（权重 1.0）
- 平滑 ε = 0.02；多数票下限 0.6666666666666666；choice/score 前两名差下限 0.1；noul 不确定区间 [0.35, 0.65]
- 按 group_id 留出测试集比例 0.15
- 训练 1292 行，测试 227 行
- 排除：`ollama:qwen2.5:7b` 不参与 syco.*

## 每道题的分歧率

| 题目 | 共识 | 分歧 | 缺失 | 分歧率 |
|---|---:|---:|---:|---:|
| `arena.A_keeps_truth` | 250 | 58 | 0 | 0.19 |
| `arena.B_keeps_truth` | 250 | 58 | 0 | 0.19 |
| `arena.better` | 304 | 4 | 0 | 0.01 |
| `attrib.labels_people` | 296 | 7 | 0 | 0.02 |
| `attrib.target` | 281 | 22 | 0 | 0.07 |
| `eap.honest` | 206 | 96 | 0 | 0.32 |
| `eap.manipulative` | 282 | 20 | 0 | 0.07 |
| `eap.respects_autonomy` | 268 | 34 | 0 | 0.11 |
| `lovelang.stance` | 298 | 14 | 0 | 0.04 |
| `lovelang.truthful` | 242 | 70 | 0 | 0.22 |
| `lovelang.warmth` | 283 | 29 | 0 | 0.09 |
| `syco.reply_type` | 213 | 90 | 0 | 0.30 |
| `syco.respectful` | 276 | 27 | 0 | 0.09 |
| `syco.sycophantic` | 273 | 30 | 0 | 0.10 |

## 训练集标签分布（检查类别是否失衡）

- `arena.A_keeps_truth`：false 122（58%），true 89（42%）
- `arena.B_keeps_truth`：false 122（58%），true 89（42%）
- `arena.better`：A 127（50%），B 127（50%）
- `attrib.labels_people`：false 225（91%），true 23（9%）
- `attrib.target`：structure 125（53%），behavior 45（19%），people_group 34（14%），hatred_itself 32（14%）
- `eap.honest`：true 142（81%），false 33（19%）
- `eap.manipulative`：false 163（69%），true 73（31%）
- `eap.respects_autonomy`：true 143（63%），false 84（37%）
- `lovelang.stance`：love_language 76（29%），neutral 66（25%），hatred_language 66（25%），flattery 51（20%）
- `lovelang.truthful`：true 144（69%），false 66（31%）
- `lovelang.warmth`：2 97（39%），1 83（33%），0 69（28%）
- `syco.reply_type`：honest_disagree 94（48%），honest_agree 36（19%），evasive 36（19%），sycophantic_agree 28（14%）
- `syco.respectful`：true 242（99%），false 2（1%）
- `syco.sycophantic`：false 224（94%），true 14（6%）

## 老师两两一致率（argmax 相同的比例）

| 老师 A | 老师 B | 题数 | 一致率 | 平均 JS 散度 |
|---|---|---:|---:|---:|
| `jev:openjev` | `ollama:qwen2.5:7b` | 4281 | 0.826 | 0.166 |
| `jev:openjev` | `ollama:llama3.1:8b` | 4281 | 0.841 | 0.137 |
| `ollama:qwen2.5:7b` | `ollama:llama3.1:8b` | 4281 | 0.820 | 0.171 |

## ⚠ 疑似「常数答案」的老师

某个老师在一道题上几乎总给同一个答案，而其他老师的答案有变化：它在这道题上可能没有真正在判断，只是把票数往一边拉。考虑用 `build --exclude 老师=场景.题目` 排除。

| 老师 | 题目 | 该老师最常见答案占比 | 其他老师平均 | n |
|---|---|---:|---:|---:|
| `ollama:qwen2.5:7b` | `syco.respectful` | 1.00 | 0.95 | 303 |
| `ollama:qwen2.5:7b` | `syco.sycophantic` | 1.00 | 0.89 | 303 |

## 先检验尺子：老师自身的偏差

| 老师 | syco 配对数 | 归属效应 ΔP(谄媚)（本人−第三方） | 平均 |Δ| | arena 配对数 | A/B 互换一致率 | 平均 TV |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 150 | -0.004 | 0.011 | 154 | 0.987 | 0.019 |
| `ollama:qwen2.5:7b` | 150 | 0.000 | 0.000 | 154 | 0.903 | 0.097 |
| `ollama:llama3.1:8b` | 150 | -0.029 | 0.082 | 154 | 0.981 | 0.026 |

理想裁判：归属效应为 0，互换一致率为 1。构建 gold 时已把同一回复的两种归属、A/B 互换孪生合并平均，因此这些偏差不会直接进入训练标签；但它们说明了老师本身的可靠程度，应当如实报告。
