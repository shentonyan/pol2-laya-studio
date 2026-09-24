# 评估报告

时间：2026-09-24T05:37:50+0800
gold：`distill_data\autogold\judge_gold_ollama_gemma4_26b.jsonl`（227 条）

> gold 来自一个没有参与训练标签的独立模型：这里的准确率回答「与独立裁判是否一致」，仍不等于判得对。

## `all`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 632 | 0 | 0.900 | 0.143 | 0.061 | 0.357 |
| `ollama:qwen2.5:7b` | 632 | 0 | 0.815 | 0.353 | 0.170 | 0.398 |
| `ollama:llama3.1:8b` | 632 | 0 | 0.878 | 0.223 | 0.081 | 0.285 |
| `base:multilingual` | 632 | 0 | 0.491 | 0.810 | 0.342 | 0.719 |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 632 | 0 | 0.861 | 0.208 | 0.046 | 0.435 |

## `type:choice`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 179 | 0 | 0.927 | 0.114 | 0.044 | — |
| `ollama:qwen2.5:7b` | 179 | 0 | 0.737 | 0.521 | 0.251 | — |
| `ollama:llama3.1:8b` | 179 | 0 | 0.810 | 0.345 | 0.136 | — |
| `base:multilingual` | 179 | 0 | 0.570 | 0.607 | 0.162 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 179 | 0 | 0.804 | 0.277 | 0.065 | — |

## `type:noul`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 412 | 0 | 0.905 | 0.137 | 0.080 | — |
| `ollama:qwen2.5:7b` | 412 | 0 | 0.867 | 0.244 | 0.121 | — |
| `ollama:llama3.1:8b` | 412 | 0 | 0.920 | 0.147 | 0.068 | — |
| `base:multilingual` | 412 | 0 | 0.476 | 0.890 | 0.426 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 412 | 0 | 0.910 | 0.152 | 0.060 | — |

## `type:score`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 41 | 0 | 0.732 | 0.334 | 0.226 | 0.357 |
| `ollama:qwen2.5:7b` | 41 | 0 | 0.634 | 0.710 | 0.325 | 0.398 |
| `ollama:llama3.1:8b` | 41 | 0 | 0.756 | 0.450 | 0.163 | 0.285 |
| `base:multilingual` | 41 | 0 | 0.293 | 0.902 | 0.408 | 0.719 |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 41 | 0 | 0.610 | 0.460 | 0.128 | 0.435 |

## `arena.A_keeps_truth`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 52 | 0 | 0.731 | 0.269 | 0.118 | — |
| `ollama:qwen2.5:7b` | 52 | 0 | 0.808 | 0.350 | 0.167 | — |
| `ollama:llama3.1:8b` | 52 | 0 | 0.942 | 0.085 | 0.026 | — |
| `base:multilingual` | 52 | 0 | 0.538 | 0.753 | 0.341 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 52 | 0 | 0.962 | 0.116 | 0.160 | — |

## `arena.B_keeps_truth`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 52 | 0 | 0.827 | 0.224 | 0.115 | — |
| `ollama:qwen2.5:7b` | 52 | 0 | 0.750 | 0.436 | 0.231 | — |
| `ollama:llama3.1:8b` | 52 | 0 | 0.962 | 0.090 | 0.058 | — |
| `base:multilingual` | 52 | 0 | 0.500 | 0.754 | 0.359 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 52 | 0 | 0.923 | 0.140 | 0.086 | — |

## `arena.better`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 52 | 0 | 0.981 | 0.050 | 0.042 | — |
| `ollama:qwen2.5:7b` | 52 | 0 | 0.923 | 0.154 | 0.077 | — |
| `ollama:llama3.1:8b` | 52 | 0 | 0.981 | 0.047 | 0.032 | — |
| `base:multilingual` | 52 | 0 | 0.558 | 0.579 | 0.191 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 52 | 0 | 0.981 | 0.046 | 0.028 | — |

## `attrib.labels_people`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 49 | 0 | 0.980 | 0.092 | 0.118 | — |
| `ollama:qwen2.5:7b` | 49 | 0 | 0.939 | 0.122 | 0.061 | — |
| `ollama:llama3.1:8b` | 49 | 0 | 0.959 | 0.086 | 0.048 | — |
| `base:multilingual` | 49 | 0 | 0.837 | 0.240 | 0.130 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 49 | 0 | 0.959 | 0.096 | 0.056 | — |

## `attrib.target`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 49 | 0 | 0.878 | 0.191 | 0.100 | — |
| `ollama:qwen2.5:7b` | 49 | 0 | 0.735 | 0.503 | 0.245 | — |
| `ollama:llama3.1:8b` | 49 | 0 | 0.755 | 0.463 | 0.204 | — |
| `base:multilingual` | 49 | 0 | 0.510 | 0.740 | 0.243 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 49 | 0 | 0.776 | 0.329 | 0.111 | — |

## `eap.honest`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 48 | 0 | 0.812 | 0.193 | 0.117 | — |
| `ollama:qwen2.5:7b` | 48 | 0 | 0.812 | 0.352 | 0.188 | — |
| `ollama:llama3.1:8b` | 48 | 0 | 0.792 | 0.394 | 0.208 | — |
| `base:multilingual` | 48 | 0 | 0.292 | 1.312 | 0.685 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 48 | 0 | 0.833 | 0.256 | 0.124 | — |

## `eap.manipulative`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 48 | 0 | 0.979 | 0.037 | 0.051 | — |
| `ollama:qwen2.5:7b` | 48 | 0 | 0.958 | 0.065 | 0.028 | — |
| `ollama:llama3.1:8b` | 48 | 0 | 0.979 | 0.023 | 0.007 | — |
| `base:multilingual` | 48 | 0 | 0.625 | 0.717 | 0.352 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 48 | 0 | 0.917 | 0.134 | 0.063 | — |

## `eap.respects_autonomy`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 48 | 0 | 0.938 | 0.102 | 0.111 | — |
| `ollama:qwen2.5:7b` | 48 | 0 | 0.896 | 0.190 | 0.111 | — |
| `ollama:llama3.1:8b` | 48 | 0 | 0.917 | 0.144 | 0.083 | — |
| `base:multilingual` | 48 | 0 | 0.521 | 0.705 | 0.320 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 48 | 0 | 0.854 | 0.205 | 0.075 | — |

## `lovelang.stance`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 41 | 0 | 0.878 | 0.168 | 0.075 | — |
| `ollama:qwen2.5:7b` | 41 | 0 | 0.829 | 0.352 | 0.187 | — |
| `ollama:llama3.1:8b` | 41 | 0 | 0.805 | 0.374 | 0.171 | — |
| `base:multilingual` | 41 | 0 | 0.585 | 0.574 | 0.147 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 41 | 0 | 0.732 | 0.362 | 0.184 | — |

## `lovelang.truthful`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 41 | 0 | 0.951 | 0.156 | 0.181 | — |
| `ollama:qwen2.5:7b` | 41 | 0 | 0.756 | 0.450 | 0.203 | — |
| `ollama:llama3.1:8b` | 41 | 0 | 0.902 | 0.195 | 0.098 | — |
| `base:multilingual` | 41 | 0 | 0.195 | 1.419 | 0.733 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 41 | 0 | 0.829 | 0.210 | 0.141 | — |

## `lovelang.warmth`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 41 | 0 | 0.732 | 0.334 | 0.226 | 0.357 |
| `ollama:qwen2.5:7b` | 41 | 0 | 0.634 | 0.710 | 0.325 | 0.398 |
| `ollama:llama3.1:8b` | 41 | 0 | 0.756 | 0.450 | 0.163 | 0.285 |
| `base:multilingual` | 41 | 0 | 0.293 | 0.902 | 0.408 | 0.719 |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 41 | 0 | 0.610 | 0.460 | 0.128 | 0.435 |

## `syco.reply_type`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 37 | 0 | 0.973 | 0.044 | 0.049 | — |
| `ollama:qwen2.5:7b` | 37 | 0 | 0.378 | 1.249 | 0.631 | — |
| `ollama:llama3.1:8b` | 37 | 0 | 0.649 | 0.577 | 0.189 | — |
| `base:multilingual` | 37 | 0 | 0.649 | 0.505 | 0.247 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 37 | 0 | 0.676 | 0.440 | 0.158 | — |

## `syco.respectful`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 37 | 0 | 1.000 | 0.079 | 0.175 | — |
| `ollama:qwen2.5:7b` | 37 | 0 | 1.000 | 0.000 | 0.000 | — |
| `ollama:llama3.1:8b` | 37 | 0 | 0.865 | 0.204 | 0.036 | — |
| `base:multilingual` | 37 | 0 | 0.000 | 1.875 | 0.968 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 37 | 0 | 1.000 | 0.059 | 0.163 | — |

## `syco.sycophantic`

| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|---:|
| `jev:openjev` | 37 | 0 | 1.000 | 0.028 | 0.098 | — |
| `ollama:qwen2.5:7b` | 37 | 0 | 0.919 | 0.162 | 0.081 | — |
| `ollama:llama3.1:8b` | 37 | 0 | 0.946 | 0.126 | 0.081 | — |
| `base:multilingual` | 37 | 0 | 0.649 | 0.475 | 0.294 | — |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 37 | 0 | 0.919 | 0.145 | 0.068 | — |

## 不变性诊断

| 系统 | syco 配对 | 归属效应 ΔP(谄媚) | 平均 |Δ| | arena 配对 | A/B 互换一致率 |
|---|---:|---:|---:|---:|---:|
| `jev:openjev` | 18 | -0.006 | 0.010 | 26 | 1.000 |
| `ollama:qwen2.5:7b` | 18 | 0.000 | 0.000 | 26 | 0.962 |
| `ollama:llama3.1:8b` | 18 | -0.093 | 0.093 | 26 | 1.000 |
| `base:multilingual` | 18 | 0.118 | 0.118 | 26 | 0.115 |
| `C:\Users\Administrator\pol2-laya-studio\distill_data\models\laya-pol2-20260924-0504` | 18 | -0.006 | 0.023 | 26 | 1.000 |
