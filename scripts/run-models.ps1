<#
.SYNOPSIS
Benchmark a list of models, one after the other.

.EXAMPLE
$env:API_KEY = "..."; scripts\run-models.ps1 --limit 100

.EXAMPLE
$env:API_KEY = "..."; scripts\run-models.ps1 --models other-list.txt --limit 0

.DESCRIPTION
Everything but --models is passed on to the benchmark unchanged. One model failing does not
stop the others. Runs land in results\<timestamp>\<model>.json.
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-Location (Join-Path $PSScriptRoot '..')

if (-not $env:API_KEY)
{
    Write-Error "API_KEY is not set"
    exit 1
}

# Read out --models, pass everything else on. Done by hand because a declared parameter
# would swallow the first benchmark flag as a positional argument.
$Models = Join-Path $PSScriptRoot 'models.txt'
$forward = @()
for ($i = 0; $i -lt $Rest.Count; $i++) {
    if ($Rest[$i] -in '--models', '-Models')
    {
        $Models = $Rest[++$i]
        if (-not $Models)
        {
            Write-Error "--models needs a file"; exit 1
        }
    }
    else
    {
        $forward += $Rest[$i]
    }
}

$ids = Get-Content $Models |
        ForEach-Object { ($_ -replace '#.*', '').Trim() } |
        Where-Object { $_ }

if (-not $ids)
{
    Write-Error "no models listed in $Models"
    exit 1
}

#cargo build --release
#if ($LASTEXITCODE -ne 0)
#{
#    exit 1
#}
$binary = Join-Path 'target' 'release' 'kolla-benchmark.exe'

$batch = Join-Path 'results' (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Force -Path $batch | Out-Null

# The file a model writes to, its name stripped of anything a path would not like.
function Get-ResultPath($model)
{
    Join-Path $batch (($model -replace '[^a-zA-Z0-9.\-]', '_') + '.json')
}

foreach ($model in $ids)
{
    Write-Output ""
    Write-Output "=== $model ==="
    & $binary --model $model --output (Get-ResultPath $model) @forward
    if ($LASTEXITCODE -ne 0)
    {
        Write-Warning "$model failed, continuing"
    }
}

# Pull the metrics back out of the written runs.
$summary = foreach ($model in $ids)
{
    $result = Get-ResultPath $model
    $metrics = if (Test-Path $result)
    {
        (Get-Content $result -Raw | ConvertFrom-Json).metrics
    }
    # Invariant culture, so the numbers read the same everywhere.
    $number = { param($value) $value.ToString('F4', [cultureinfo]::InvariantCulture) }
    [pscustomobject]@{
        Model = $model
        Precision = if ($metrics)
        {
            & $number $metrics.precision
        }
        else
        {
            'no result'
        }
        Recall = if ($metrics)
        {
            & $number $metrics.recall
        }
        else
        {
            ''
        }
        F05 = if ($metrics)
        {
            & $number $metrics.f05
        }
        else
        {
            ''
        }
    }
}

Write-Output ""
$summary | Format-Table -AutoSize
Write-Output "runs written to $batch"
