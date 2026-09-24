# 不依赖人工的独立评估：构造 gold（谄媚）+ 独立裁判 gold（gemma4:26b）
# 用法：在仓库根目录运行  .\scripts\run_autogold.ps1
# 可重复运行：已生成的样本和已打的标签会跳过。

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "$HOME\laya-env\Scripts\python.exe"

$model = Get-ChildItem distill_data\models -Directory -Filter "laya-pol2-*" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $model) { throw "找不到 distill_data\models\laya-pol2-* ，请先运行 train。" }
$student = $model.FullName
$fresh = "distill_data\autogold\syco_fresh.jsonl"

function Step($title, [scriptblock]$cmd) {
    Write-Host ""
    Write-Host "==== $title ====" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "步骤失败：$title（退出码 $LASTEXITCODE）" }
}

Write-Host "学生模型：$student"

if (Test-Path $fresh) {
    Write-Host "已存在 $fresh ，跳过生成。要重新生成请先删除这个文件。"
} else {
    Step "1/7 生成全新谄媚样本（gemma4:26b，训练目录之外）" {
        & $py -m distill generate --model gemma4:26b --only syco --per-scenario 300 --seed 7 --out $fresh
    }
}

Step "2/7 按构造生成 gold" {
    & $py -m distill autogold design --fresh $fresh
}

Step "3/7 Jev 标注新样本" {
    & $py -m distill label --teacher jev --workers 6 --max-retries 8 --states $fresh
}

Step "4/7 llama3.1:8b 标注新样本" {
    & $py -m distill label --teacher ollama:llama3.1:8b --states $fresh
}

Step "5/7 评估：构造 gold" {
    & $py -m distill eval --gold distill_data\autogold\design_gold_fresh.jsonl `
        --model base:multilingual --model $student `
        --teacher jev --teacher ollama:llama3.1:8b `
        --states $fresh --out distill_data\eval\eval_design_fresh.md
}

Step "6/7 独立裁判 gemma4:26b 标注测试集" {
    & $py -m distill autogold judge --judge ollama:gemma4:26b `
        --used jev --used ollama:qwen2.5:7b --used ollama:llama3.1:8b
}

Step "7/7 评估：独立裁判 gold" {
    & $py -m distill eval --gold distill_data\autogold\judge_gold_ollama_gemma4_26b.jsonl `
        --model base:multilingual --model $student `
        --teacher jev --teacher ollama:qwen2.5:7b --teacher ollama:llama3.1:8b `
        --out distill_data\eval\eval_judge_gemma4.md
}

Write-Host ""
Write-Host "全部完成。报告：" -ForegroundColor Green
Write-Host "  distill_data\eval\eval_design_fresh.md"
Write-Host "  distill_data\eval\eval_judge_gemma4.md"
