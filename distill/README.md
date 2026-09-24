# distill · 用 Jev 和本地 LLM 当老师，蒸馏 PoL2 专用的 Laya 判别模型

基础 Laya checkpoint 在 PoL2 场景上是零样本，官方基准里这种用法接近随机。这个目录要做的，是把它变成一个**在 PoL2 场景上专门训练过、并且接受过诚实评估的学生模型**：

```
scenarios.json 的问题
  → 种子 + 合成 state（本地 Ollama 生成）
  → 多个老师打软标签（Jev + 本地 LLM 裁判）
  → 聚合为共识；老师有分歧的题目交人工复核，不进训练
  → Laya RLCD 微调（单显卡）+ 温度校准
  → 在盲标的人工金标集上，比较基础模型 / 学生 / 各个老师
```

打标签、聚合、人工标注这几步只用 Python 标准库；训练和评估需要 Studio 本身的环境（torch + laya）。

## 诚实条款

1. **学生的上限就是老师。**gold 是老师共识，不是真值；老师的偏差会被一起学走。
2. **只有人工金标集能回答「判得对不对」。**`test.jsonl` 上的准确率只说明学生「学得像不像老师」。
3. **先检验尺子。**`report.md` 会报告每个老师在两个对称条件下的偏差：
   - 谄媚的第三方条件：同一回复只改变归属时，P(谄媚) 变了多少；
   - 爱语擂台：A/B 互换后判断是否跟着互换。
4. **负结果照样报告。**如果学生在人工金标集上没有超过基础模型，这个结果本身就有价值。
5. **Jev 的使用条款要先查清楚。**有些 API 不允许用输出训练其他模型；在 OpenJEV 控制台或向 TypeSafe 确认之前，可以只用本地 LLM 当老师（去掉 `--teacher jev` 即可）。
6. **state 会经网络发给 Jev。**只放合成文本，不要放真实个人的对话或隐私信息。

## 设计要点

| 做法 | 原因 |
|---|---|
| 按 `group_id` 切分训练/测试 | 同一主张的不同归属、同一对改写的 A/B 互换，必须落在同一侧，否则测试集会泄漏 |
| 爱语擂台自动生成 A/B 互换孪生，gold 合并后再镜像回去 | 抵消老师的位置偏差，也让学生没法靠位置作答 |
| 谄媚测试：同一主张 × 2 种归属 × 3 种回复；同一回复在两种归属下的 gold 合并平均 | 「回复是否违背证据地附和」这个事实判断不应随归属变化。老师的归属偏差不会进入标签，但会在报告里如实列出 |
| 本地 LLM 用 JSON Schema 约束解码，温度采样多次投票 | 答案一定合法；分布是模型自身的输出波动，**不是**校准过的概率 |
| 每个老师先做 ε 平滑再混合 | Jev 有时给真实标签 0 概率，平滑后不会把学生推向极端 |
| 分歧题目不进训练 | 多数票比例低于 2/3、noul 共识落在 [0.35, 0.65]、choice/score 前两名差小于 0.1 时，送去人工复核 |
| 生成模型和老师模型分开 | 避免老师偏爱「自己写的」文本。建议用 gemma4 生成，用 Jev + qwen2.5 + llama3.1 打标签 |
| 默认用 `laya-multilingual` 做基座 | PoL2 文本以中文为主；英文 checkpoint 遇到非拉丁文字会出错，而且错的时候依然很自信 |

## 使用（Windows PowerShell）

在 Studio 仓库根目录执行，沿用 Studio 的虚拟环境：

```powershell
cd ~\pol2-laya-studio
$py = ".\.venv\Scripts\python.exe"      # 如果用的是 ~\laya-env，就改成 "~\laya-env\Scripts\python.exe"

# 0. 离线自检：不需要 key、网络和显卡
& $py -m pip install pytest
& $py -m pytest tests -q

# 1. Jev 的 key 放进 .env（已被 .gitignore 排除），不要贴进聊天或提交记录
Copy-Item .env.example .env
notepad .env
```

### 第一步：准备 state

```powershell
& $py -m distill seeds                                              # scenarios.json 里的样例 → 16 条种子
& $py -m distill generate --model gemma4:26b --per-scenario 300     # 每个 PoL2 场景约 300 条合成 state
```

生成的文件在 `distill_data\states\`。每行一条，可以直接打开检查，删掉质量差的行即可。`--only lovelang,syco` 可以只处理部分场景。

### 第二步：老师打标签（可随时中断，重跑会续上）

```powershell
# 先各试 20 条，确认 key、Ollama 和输出格式都没问题
& $py -m distill label --teacher jev --limit 20
& $py -m distill label --teacher ollama:qwen2.5:7b --limit 20

# 然后全量跑
& $py -m distill label --teacher jev --workers 3
& $py -m distill label --teacher ollama:qwen2.5:7b --teacher ollama:llama3.1:8b --samples 3
```

结果按老师分别存在 `distill_data\labels\*.jsonl`。失败记录在 `errors.jsonl`；连续失败 5 次会自动停止。

### 第三步：聚合成训练数据

```powershell
& $py -m distill build --teacher jev --teacher ollama:qwen2.5:7b --teacher ollama:llama3.1:8b
```

输出到 `distill_data\dataset\`：

- `train.jsonl` / `test.jsonl`：格式与 Laya 官方 notebook 读取的 `LocalLLaMA/typed-decisions` 相同，也可以直接拿去 Kaggle 训练。
- `review.csv`：老师有分歧的题目，用 Excel 打开。
- `report.md`：**先读这个。**包括每道题的分歧率、训练集的标签分布（检查类别是否失衡），以及老师自身的偏差诊断。

可选参数：`--weight jev:openjev=1.5` 调整老师权重；`--agree-min`、`--margin-min` 调整分歧门槛；`--no-tie-syco` 关闭归属合并。

### 第四步：人工金标集（盲标）

```powershell
& $py -m distill annotate export --per-scenario 40
# 用 Excel 打开 distill_data\annotation\to_label.csv，在 human_label 列填写：
#   choice 填选项名；score 填 0/1/2…；noul 填 true/false（也可以填 是/否）
# 另存为 to_label_你的名字.csv；可以请别人各填一份
& $py -m distill annotate import --csv distill_data\annotation\to_label_shenton.csv
```

表格里不显示任何老师或模型的答案，以免人被锚定。导入多份时会报告人与人之间的一致率。

### 第五步：微调（单显卡）

```powershell
& $py -m distill train                         # 默认基座：convaiinnovations/laya 的 multilingual
& $py -m distill train --subfolder typed-decisions --epochs 3   # 也可以从官方微调版继续训练
```

默认每次前向 8 条、累积 8 步、训练 4 个 epoch，并开启梯度检查点。显存不够时用 `--micro-batch 4 --grad-accum 16`。输出在 `distill_data\models\laya-pol2-<时间>\`，可以直接用 `laya.Agent("<目录>")` 加载。每个 epoch 结束都会覆盖保存一次 `checkpoint_latest\`，中途崩溃不会丢掉全部进度。

### 第六步：评估

```powershell
& $py -m distill eval --gold distill_data\annotation\human_gold.jsonl `
    --model base:multilingual --model distill_data\models\laya-pol2-20260924-1200 `
    --teacher jev --teacher ollama:qwen2.5:7b --teacher ollama:llama3.1:8b
```

报告写在 `distill_data\eval\`，包含整体、按题型、按题目的准确率、Brier、ECE、score MAE，以及每个系统的不变性诊断（归属效应、A/B 互换一致率）。

## 目录

```
distill/common.py     路径、JSONL、分布的规范化 / 平滑 / 聚合
distill/teachers.py   Jev、Ollama、Mock 三种老师
distill/states.py     种子导出、合成生成、结构性孪生（A/B 互换、归属配对）
distill/label.py      可续跑的批量打标签
distill/build.py      聚合、分歧判定、按组切分、老师偏差诊断、报告
distill/annotate.py   盲标导出 / 导入
distill/train.py      单卡 RLCD 微调（改写自官方 Kaggle notebook）
distill/evaluate.py   基础模型 / 学生 / 老师的对比评估
tests/test_distill.py 用假的 Ollama / OpenJEV 服务跑完整流程
distill_data/         数据与模型（models/ 和 .env 不进 git）
```
