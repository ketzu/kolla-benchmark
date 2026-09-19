<#
.SYNOPSIS
Benchmark a list of models, one after the other, optionally against a list of prompts.

.EXAMPLE
$env:API_KEY = "..."; scripts\run-models.ps1 --limit 100

.EXAMPLE
$env:API_KEY = "..."; scripts\run-models.ps1 --models other-list.txt --limit 0

.EXAMPLE
$env:API_KEY = "..."; scripts\run-models.ps1 --prompts scripts\prompts.json --limit 100

.EXAMPLE
$env:API_KEY = "..."; scripts\run-models.ps1 --results-dir iterate-results --iterate --limit 100

.DESCRIPTION
Everything but --models, --prompts and --results-dir is passed on to the benchmark unchanged. One
run failing does not stop the others. Runs land in results\<timestamp>\<model>.json, or below the
directory given with --results-dir.

With --prompts every model runs against every prompt of a JSON file: an array of objects with a
"user" template containing {sentence}, an optional "system" prompt and an optional "name". Prompts
are numbered from 1 in file order, and runs land in results\<timestamp>\<model>\p<number>.json.
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

# Read out --models, --prompts and --results-dir, pass everything else on. Done by hand because a
# declared parameter would swallow the first benchmark flag as a positional argument.
$Models = Join-Path $PSScriptRoot 'models.txt'
$PromptFile = $null
$ResultsDir = 'results'
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
    elseif ($Rest[$i] -in '--prompts', '-Prompts')
    {
        $PromptFile = $Rest[++$i]
        if (-not $PromptFile)
        {
            Write-Error "--prompts needs a file"; exit 1
        }
    }
    elseif ($Rest[$i] -in '--results-dir', '-ResultsDir')
    {
        $ResultsDir = $Rest[++$i]
        if (-not $ResultsDir)
        {
            Write-Error "--results-dir needs a directory"; exit 1
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

# Checked up front, so that a broken prompt does not surface only after hours of other runs.
$prompts = @()
if ($PromptFile)
{
    if ($forward | Where-Object { $_ -in '--prompt', '--system' })
    {
        Write-Error "--prompt and --system cannot be combined with --prompts"; exit 1
    }
    try
    {
        $prompts = @(Get-Content $PromptFile -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop)
    }
    catch
    {
        Write-Error "reading ${PromptFile}: $_"; exit 1
    }
    $valid = $prompts.Count -gt 0
    foreach ($prompt in $prompts)
    {
        $valid = $valid -and $prompt -is [pscustomobject] -and
                $prompt.user -is [string] -and $prompt.user.Contains('{sentence}') -and
                ($null -eq $prompt.PSObject.Properties['system'] -or $prompt.system -is [string]) -and
                ($null -eq $prompt.PSObject.Properties['name'] -or $prompt.name -is [string])
    }
    if (-not $valid)
    {
        Write-Error "$PromptFile must be a non-empty array of {`"name`"?: ..., `"user`": ..., `"system`"?: ...} with {sentence} in every user template"
        exit 1
    }
}

#cargo build --release
#if ($LASTEXITCODE -ne 0)
#{
#    exit 1
#}
$binary = Join-Path 'target' 'release' 'kolla-benchmark.exe'

$batch = Join-Path $ResultsDir (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Force -Path $batch | Out-Null

# A model name stripped of anything a path would not like.
function Get-SafeName($model)
{
    $model -replace '[^a-zA-Z0-9.\-]', '_'
}

# Every run: its model, its prompt number, the flags that send the prompt, and its result file.
$runs = foreach ($model in $ids)
{
    if (-not $PromptFile)
    {
        [pscustomobject]@{
            Model = $model
            Prompt = $null
            Flags = @()
            Result = Join-Path $batch ((Get-SafeName $model) + '.json')
        }
        continue
    }
    for ($n = 0; $n -lt $prompts.Count; $n++) {
        $flags = @('--prompt', $prompts[$n].user)
        $type = 'user'
        if ($null -ne $prompts[$n].system)
        {
            $flags = @('--system', $prompts[$n].system) + $flags
            $type = 'system+user'
        }
        $name = if ($prompts[$n].name) { $prompts[$n].name } else { 'unnamed' }
        [pscustomobject]@{
            Model = $model
            Prompt = "p$( $n + 1 ) $name/$type"
            Flags = $flags
            Result = Join-Path $batch (Get-SafeName $model) "p$( $n + 1 ).json"
        }
    }
}

foreach ($run in $runs)
{
    $label = if ($run.Prompt)
    {
        "$( $run.Model ), $( $run.Prompt )"
    }
    else
    {
        $run.Model
    }
    Write-Output ""
    Write-Output "=== $label ==="
    $flags = $run.Flags
    & $binary --model $run.Model --output $run.Result @flags @forward
    if ($LASTEXITCODE -ne 0)
    {
        Write-Warning "$label failed, continuing"
    }
}

# Pull the metrics back out of the written runs.
# Invariant culture, so the numbers read the same everywhere.
$number = { param($value) $value.ToString('F4', [cultureinfo]::InvariantCulture) }
$summary = foreach ($run in $runs)
{
    $metrics = if (Test-Path $run.Result)
    {
        (Get-Content $run.Result -Raw | ConvertFrom-Json).metrics
    }
    $row = [ordered]@{ Model = $run.Model }
    if ($run.Prompt)
    {
        $row.Prompt = $run.Prompt
    }
    $row.Precision = if ($metrics)
    {
        & $number $metrics.precision
    }
    else
    {
        'no result'
    }
    $row.Recall = if ($metrics)
    {
        & $number $metrics.recall
    }
    else
    {
        ''
    }
    $row.F05 = if ($metrics)
    {
        & $number $metrics.f05
    }
    else
    {
        ''
    }
    [pscustomobject]$row
}

Write-Output ""
$summary | Format-Table -AutoSize
Write-Output "runs written to $batch"
