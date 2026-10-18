param(
    [ValidateSet('ci', 'demo', 'serve', 'worker')]
    [string]$Mode = 'ci'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([System.IO.Path]::GetPathRoot($projectRoot) -ne 'D:\') {
    throw 'Keep this project and its runtime files on D:.'
}
Set-Location -LiteralPath $projectRoot
$tempPath = Join-Path $projectRoot 'outputs\temp'
New-Item -ItemType Directory -Path $tempPath -Force | Out-Null
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:PIP_CACHE_DIR = Join-Path $projectRoot 'outputs\pip-cache'
$env:HF_HOME = Join-Path $projectRoot 'outputs\model-cache\huggingface'
$env:TORCH_HOME = Join-Path $projectRoot 'outputs\model-cache\torch'
$env:TIKTOKEN_CACHE_DIR = Join-Path $projectRoot 'outputs\model-cache\tiktoken'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
if (-not $env:MY_RAG_CONFIG) { $env:MY_RAG_CONFIG = Join-Path $projectRoot 'config.yaml' }
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
switch ($Mode) {
    'ci' {
        $env:PYTEST_ADDOPTS = '-o cache_dir=outputs/pytest-cache --basetemp=outputs/pytest-run-ci'
        & $pythonPath -u tools/ci.py
    }
    'demo' { & $pythonPath -u -m agent_runtime.demo }
    'serve' { & $pythonPath -u -m agent_runtime serve }
    'worker' { & $pythonPath -u -m agent_runtime.worker }
}
exit $LASTEXITCODE
