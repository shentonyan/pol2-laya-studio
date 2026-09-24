# 评估报告

时间：2026-09-24T05:29:58+0800
gold：`distill_data\autogold\design_gold_fresh.jsonl`（300 条）

> gold 按构造得到（主张真假 × 回复类型），不来自任何模型或人工判断。局限：生成模型写错回复类型时，gold 也会跟着错。

## `all`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 564 | 0 | 0.929 | 0.121 | 0.047 | — |
| `ollama:llama3.1:8b` | 564 | 0 | 0.754 | 0.405 | 0.120 | — |
| `base:multilingual` | 564 | 0 | 0.638 | 0.512 | 0.130 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 564 | 0 | 0.770 | 0.346 | 0.055 | — |

## `type:choice`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 264 | 0 | 0.924 | 0.122 | 0.048 | — |
| `ollama:llama3.1:8b` | 264 | 0 | 0.621 | 0.630 | 0.246 | — |
| `base:multilingual` | 264 | 0 | 0.549 | 0.590 | 0.051 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 264 | 0 | 0.750 | 0.371 | 0.075 | — |

## `type:noul`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 300 | 0 | 0.933 | 0.120 | 0.088 | — |
| `ollama:llama3.1:8b` | 300 | 0 | 0.870 | 0.207 | 0.051 | — |
| `base:multilingual` | 300 | 0 | 0.717 | 0.444 | 0.208 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 300 | 0 | 0.787 | 0.323 | 0.093 | — |

## `syco.reply_type`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 264 | 0 | 0.924 | 0.122 | 0.048 | — |
| `ollama:llama3.1:8b` | 264 | 0 | 0.621 | 0.630 | 0.246 | — |
| `base:multilingual` | 264 | 0 | 0.549 | 0.590 | 0.051 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 264 | 0 | 0.750 | 0.371 | 0.075 | — |

## `syco.sycophantic`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 300 | 0 | 0.933 | 0.120 | 0.088 | — |
| `ollama:llama3.1:8b` | 300 | 0 | 0.870 | 0.207 | 0.051 | — |
| `base:multilingual` | 300 | 0 | 0.717 | 0.444 | 0.208 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 300 | 0 | 0.787 | 0.323 | 0.093 | — |

## 不变性诊断

| 系统 | syco 配对 | 归属效应 ΔP(谄媚) | 平均 |Δ| | arena 配对 | A/B 互换一致率 |
|---|---:|---:|---:|---:|---:|
| `jev:openjev` | 150 | -0.005 | 0.014 | 0 | — |
| `ollama:llama3.1:8b` | 150 | -0.044 | 0.084 | 0 | — |
| `base:multilingual` | 150 | 0.028 | 0.067 | 0 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 150 | 0.006 | 0.019 | 0 | — |
