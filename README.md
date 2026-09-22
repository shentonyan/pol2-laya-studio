# PoL2 · Laya Studio

用开源的 [Laya](https://github.com/NandhaKishorM/laya) 类型化决策模型，在本地做**毫秒级**的结构化判断，并在一个交互页面里可视化结果。场景分两组：通用 Agent 场景，以及 [PoL2（爱的证明）](https://github.com/naturaldao/NaturalDAO) 相关的实验场景。

A local, interactive playground for the open-weight **Laya** typed-decision models, with general agent scenarios and scenarios from the **PoL2 (Proof of Love)** project. [English below](#english).

**在线演示 / Live demo：** `https://<你的用户名>.github.io/pol2-laya-studio/`（开启 GitHub Pages 后可用，见下文）

---

## 这是什么

Laya 不生成文本。给它一段**状态**（文本或 JSON）和一组**类型化问题**，它一次前向传播直接返回带概率的答案：

| 类型 | 含义 | 返回 |
|---|---|---|
| `choice` | 从命名选项中选一个 | 每个选项的概率 |
| `score` | 在有序等级上打分 | 各等级概率 + 期望分 |
| `noul` | 判断一个命题 | P(true) |

本仓库在它外面包了一层界面：场景选择、样例切换、可编辑的问题 JSON、三模型并排对比、批量评测和 CSV 导出。

> 本项目与 Laya 的作者 Convai Innovations 没有隶属关系，也不使用 TypeSafe Jev 或任何云端 API。所有推理都在你自己的机器上完成。

## 场景

**通用**
- 客服工单分诊：部门路由、紧急度、流失风险、是否要求退款
- Agent 提示注入护栏：网页或工具返回的内容是否试图操纵 Agent
- Agent 工具路由：一次前向传播选出下一步工具
- 论文摘要分诊：领域归类与相关度

**PoL2 · 爱的证明**
- 爱语 / 恨语辨识：关系姿态、是否诚实、关怀程度
- 谄媚检测 · 第三方条件：同一个错误主张，归属用户本人与归属第三方时，AI 回复是否不同
- 爱语擂台 · 快速裁判：两个改写版本，哪个更好地做到「说真话又有爱」
- 威胁归因检查：依据「威胁来自恨，而非人」，文本把问题归给人群、行为、恨本身还是结构
- 伦理对齐协议 · 代理检查：回复是否施压操纵、是否尊重自主、是否诚实

所有场景定义在 [`docs/scenarios.json`](docs/scenarios.json)，改这个文件就能增删场景、样例和问题。

## 诚实条款

按 PoL2 的方法论原则（检验诚实理论的方法本身必须诚实），先说清楚：

1. **基础模型零样本准确率有限。** 官方报告基础 checkpoint 在 typed-decisions 基准上接近随机水平，只有微调过的 checkpoint 才达到 0.766。本仓库的输出**只用于演示流程和快速探索，不构成任何结论的证据**。
2. **出厂概率偏过度自信。** 正式使用前，应在自己的标注数据上按问题类型重新拟合 temperature。
3. **PoL2 场景的判定标准是示例草案。** 模型判断的是文本的表面特征；合规或打分只是可验证的代理指标，不等于「爱」这一实质判准。
4. **负面结果同样要记录。** 如果 Laya 与人工或 LLM 裁判的一致率很低，这个结果本身就有价值。

## 本地运行（Windows PowerShell）

需要 Python 3.11+。有 NVIDIA 显卡时会自动用 GPU，没有也能用 CPU 跑（每次约几百毫秒）。

```powershell
git clone https://github.com/<你的用户名>/pol2-laya-studio.git
cd pol2-laya-studio
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"

# 先装 PyTorch。RTX 50 系显卡必须用 CUDA 12.8 及以上的版本：
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt

& $py app.py
```

首次运行会下载三个 checkpoint（约 2.3 GB），之后完全离线。启动完成后浏览器会自动打开 `http://127.0.0.1:7860`。

macOS / Linux 把 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python` 即可。

## 静态演示（GitHub Pages）

`docs/` 同时是前端和 GitHub Pages 的站点目录。页面打开时会先探测本地的 `app.py`：

- 找到了：**实时推理**，可以随意修改输入和问题。
- 没找到（比如在 GitHub Pages 上）：**静态演示**，回放 `docs/results.json` 里事先录制的真实结果，并在页面顶部标明录制时间和硬件。

生成录制结果：

```powershell
& $py scripts\record_demo.py     # 所有内置样例 × 4 种模型设置，写入 docs\results.json
git add docs/results.json; git commit -m "Record demo results"; git push
```

然后在 GitHub 仓库的 **Settings → Pages** 里，选择 **Deploy from a branch → main → /docs**，保存后一两分钟即可访问。

修改 `scenarios.json` 后要重新录制，否则演示页会提示找不到对应样例的结果。

## 目录结构

```
app.py                  本地服务：静态页面 + /api/predict
scripts/record_demo.py  录制静态演示用的真实结果
docs/index.html         前端（实时推理与静态演示共用）
docs/scenarios.json     场景、样例与问题定义
docs/results.json       录制结果（运行 record_demo.py 后生成）
```

## 常见问题

- **`torch.cuda.is_available()` 是 False**：先确认 PyTorch 是 CUDA 版（版本号带 `+cu128`），再检查显卡驱动；驱动升级后要重启。
- **`laya.load()` 卡住**：设置 `$env:USE_TF = "0"`（`app.py` 已默认设置）。
- **静态演示提示没有录制数据**：运行 `scripts\record_demo.py` 并提交 `docs/results.json`。

---

## English

**PoL2 · Laya Studio** is a local web playground for the open-weight [Laya](https://huggingface.co/convaiinnovations/laya) typed-decision models (choice / score / noul answers in a single forward pass). It ships general agent scenarios (ticket triage, prompt-injection guard, tool routing, abstract triage) and scenarios from the PoL2 (Proof of Love) project (love-language vs. hatred-language stance, third-party-condition sycophancy test, love-language arena judge, threat attribution, ethical-alignment-protocol proxy checks).

- `python app.py` runs live local inference (GPU if available) at `http://127.0.0.1:7860`.
- `docs/` doubles as a GitHub Pages site. Without a local backend it becomes a **static demo** that replays real outputs recorded by `scripts/record_demo.py`, clearly labelled with recording date and hardware.
- Scenarios live in `docs/scenarios.json`.

**Honesty clause.** Base Laya checkpoints are near chance on zero-shot typed decisions and ship over-confident. Outputs here demonstrate the workflow and are not evidence for any claim. PoL2 criteria are draft examples: scores are verifiable proxies, not the substantive criterion of love. Negative results should be reported.

Not affiliated with Convai Innovations or TypeSafe. Laya weights are Apache-2.0; this repository is MIT.
