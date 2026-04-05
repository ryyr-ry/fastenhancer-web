[CmdletBinding()]
param(
    [ValidateSet('tiny', 'base', 'small')]
    [string]$Model = 'small',

    [ValidateSet('scalar', 'simd')]
    [string]$Variant = 'simd',

    [string]$OutputDir,

    [string]$EmccPath,

    [switch]$FastMath,

    [switch]$WorkletBuild,

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Resolve-RepoRoot {
    return Split-Path -Parent $PSScriptRoot
}

function Resolve-EmccExecutable {
    param(
        [string]$ExplicitPath,
        [string]$RepoRoot,
        [switch]$AllowMissing
    )

    if ($ExplicitPath) {
        if (-not (Test-Path -LiteralPath $ExplicitPath)) {
            throw "Specified emcc was not found: $ExplicitPath"
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $command = Get-Command emcc -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $isWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    $emccBinary = if ($isWindows) { 'emcc.bat' } else { 'emcc' }

    if ($env:EMSDK) {
        $emsdkEmcc = Join-Path $env:EMSDK "upstream/emscripten/$emccBinary"
        if (Test-Path -LiteralPath $emsdkEmcc) {
            return $emsdkEmcc
        }
    }

    $homeDir = if ($isWindows) { $env:USERPROFILE } else { $env:HOME }
    if ($homeDir) {
        $userHomeEmcc = Join-Path $homeDir "emsdk/upstream/emscripten/$emccBinary"
        if (Test-Path -LiteralPath $userHomeEmcc) {
            return $userHomeEmcc
        }
    }

    $localEmcc = Join-Path $RepoRoot "emsdk/upstream/emscripten/$emccBinary"
    if (Test-Path -LiteralPath $localEmcc) {
        return $localEmcc
    }

    if ($AllowMissing) {
        return 'emcc'
    }

    throw 'emcc was not found. Install emsdk and set PATH or EMSDK, or pass -EmccPath.'
}

function Get-ModelConfig {
    param([string]$ModelName)

    switch ($ModelName) {
        'tiny' {
            return @{
                ModelId = 0
                InitialMemory = 1048576
                ExportName = 'FastEnhancerTiny'
                ModelDefine = 'FE_USE_TINY_48K'
            }
        }
        'base' {
            return @{
                ModelId = 1
                InitialMemory = 2097152
                ExportName = 'FastEnhancerBase'
                ModelDefine = 'FE_USE_BASE_48K'
            }
        }
        'small' {
            return @{
                ModelId = 2
                InitialMemory = 4194304
                ExportName = 'FastEnhancerSmall'
                ModelDefine = 'FE_USE_SMALL_48K'
            }
        }
        default {
            throw "Unsupported model: $ModelName"
        }
    }
}

function Format-CommandLine {
    param([string]$Executable, [string[]]$Arguments)

    $parts = @($Executable) + ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    })

    return ($parts -join ' ')
}

$repoRoot = Resolve-RepoRoot
$config = Get-ModelConfig -ModelName $Model

if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot 'dist\wasm'
}

$outputFile = Join-Path $OutputDir ("fastenhancer-{0}-{1}.js" -f $Model, $Variant)
$sourceFiles = @(
    (Join-Path $repoRoot 'src\engine\exports.c'),
    (Join-Path $repoRoot 'src\engine\fastenhancer.c'),
    (Join-Path $repoRoot 'src\engine\pipeline.c')
)
$sourceFiles += Get-ChildItem -Path (Join-Path $repoRoot 'src\engine\common') -Filter '*.c' |
    Sort-Object Name |
    ForEach-Object { $_.FullName }

$includeDirs = @(
    (Join-Path $repoRoot 'src\engine'),
    (Join-Path $repoRoot 'src\engine\common'),
    (Join-Path $repoRoot 'src\engine\configs')
)
$exportedFunctions = @(
    '_malloc',
    '_free',
    '_fe_weight_count',
    '_fe_create',
    '_fe_process_frame',
    '_fe_destroy',
    '_fe_get_hop_size',
    '_fe_set_hpf',
    '_fe_set_agc',
    '_fe_reset',
    '_fe_init',
    '_fe_process',
    '_fe_process_inplace',
    '_fe_get_input_ptr',
    '_fe_get_output_ptr',
    '_fe_get_n_fft'
) -join ','

$arguments = @()
foreach ($includeDir in $includeDirs) {
    $arguments += @('-I', $includeDir)
}
$arguments += $sourceFiles
$arguments += @(
    '--no-entry',
    '-std=c11',
    '-O3',
    '-flto',
    '-sWASM=1',
    '-sMODULARIZE=1',
    '-sEXPORT_ES6=1',
    "-sEXPORT_NAME=$($config.ExportName)",
    '-sSINGLE_FILE=1',
    '-sSINGLE_FILE_BINARY_ENCODE=1',
    '-sALLOW_MEMORY_GROWTH=0',
    "-sINITIAL_MEMORY=$($config.InitialMemory)",
    '-sSTACK_SIZE=65536',
    '-sFILESYSTEM=0',
    '-sENVIRONMENT=web,worker',
    '-sNO_EXIT_RUNTIME=1',
    '-sWASM_ASYNC_COMPILATION=1',
    '-sDYNAMIC_EXECUTION=0',
    "-sEXPORTED_FUNCTIONS=[$exportedFunctions]",
    '-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,wasmMemory',
    ('-DFE_MODEL_SIZE={0}' -f $config.ModelId),
    ('-D{0}' -f $config.ModelDefine),
    '-o', $outputFile
)

if ($Variant -eq 'simd') {
    $arguments += '-msimd128'
}

if ($FastMath) {
    $arguments += '-ffast-math'
}

$emcc = Resolve-EmccExecutable -ExplicitPath $EmccPath -RepoRoot $repoRoot -AllowMissing:$DryRun
$commandLine = Format-CommandLine -Executable $emcc -Arguments $arguments

Write-Host "repo_root=$repoRoot"
Write-Host "model=$Model"
Write-Host "variant=$Variant"
Write-Host "output=$outputFile"
Write-Host "command=$commandLine"

if ($DryRun) {
    return
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $emcc @arguments

if ($LASTEXITCODE -ne 0) {
    throw "emcc build failed. exit_code=$LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $outputFile)) {
    throw "Build completed but output file was not found: $outputFile"
}

$artifact = Get-Item -LiteralPath $outputFile
Write-Host ("built={0} bytes" -f $artifact.Length)

$jsContent = Get-Content -LiteralPath $outputFile -Raw
$forbiddenPatterns = @('setTimeout', 'setInterval')
foreach ($pattern in $forbiddenPatterns) {
    if ($jsContent -match [regex]::Escape($pattern)) {
        Write-Warning "WARNING: Output contains '$pattern' which may break AudioWorklet. Review generated JS."
    }
}

if ($WorkletBuild) {
    Write-Host "--- WorkletBuild: producing separate .wasm binary ---"

    $workletTempJs = Join-Path $OutputDir ("fastenhancer-{0}-{1}-worklet-tmp.js" -f $Model, $Variant)

    $workletArgs = @()
    foreach ($includeDir in $includeDirs) {
        $workletArgs += @('-I', $includeDir)
    }
    $workletArgs += $sourceFiles
    $workletArgs += @(
        '--no-entry',
        '-std=c11',
        '-O3',
        '-flto',
        '--profiling-funcs',
        '-sWASM=1',
        '-sMODULARIZE=1',
        '-sEXPORT_ES6=1',
        "-sEXPORT_NAME=$($config.ExportName)",
        '-sALLOW_MEMORY_GROWTH=0',
        "-sINITIAL_MEMORY=$($config.InitialMemory)",
        '-sSTACK_SIZE=65536',
        '-sFILESYSTEM=0',
        '-sENVIRONMENT=web,worker',
        '-sNO_EXIT_RUNTIME=1',
        '-sWASM_ASYNC_COMPILATION=0',
        '-sDYNAMIC_EXECUTION=0',
        "-sEXPORTED_FUNCTIONS=[$exportedFunctions]",
        '-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,wasmMemory',
        ('-DFE_MODEL_SIZE={0}' -f $config.ModelId),
        ('-D{0}' -f $config.ModelDefine),
        '-o', $workletTempJs
    )

    if ($Variant -eq 'simd') {
        $workletArgs += '-msimd128'
    }

    if ($FastMath) {
        $workletArgs += '-ffast-math'
    }

    $workletCmd = Format-CommandLine -Executable $emcc -Arguments $workletArgs
    Write-Host "worklet_command=$workletCmd"

    & $emcc @workletArgs

    if ($LASTEXITCODE -ne 0) {
        throw "emcc worklet build failed. exit_code=$LASTEXITCODE"
    }

    $workletTempWasm = [System.IO.Path]::ChangeExtension($workletTempJs, '.wasm')
    $finalWasm = Join-Path $OutputDir ("fastenhancer-{0}-{1}.wasm" -f $Model, $Variant)

    if (-not (Test-Path -LiteralPath $workletTempWasm)) {
        throw "Worklet build did not produce .wasm file: $workletTempWasm"
    }

    Move-Item -LiteralPath $workletTempWasm -Destination $finalWasm -Force

    $exportMapFile = Join-Path $OutputDir ("fastenhancer-{0}-{1}-exports.json" -f $Model, $Variant)
    if (Test-Path -LiteralPath $workletTempJs) {
        $jsGlue = Get-Content -LiteralPath $workletTempJs -Raw
        $pattern = '(?:_(?:fe_\w+|malloc|free))\s*=\s*(?:Module\[")?\w+(?:"\])?\s*=\s*wasmExports\["(\w+)"\]'
        $allMatches = [regex]::Matches($jsGlue, '((?:_fe_\w+|_malloc|_free))\s*=\s*(?:Module\["\1"\]\s*=\s*)?wasmExports\["(\w+)"\]')
        $map = @{}
        foreach ($m in $allMatches) {
            $map[$m.Groups[1].Value] = $m.Groups[2].Value
        }
        if ($map.Count -gt 0) {
            $map | ConvertTo-Json | Set-Content -LiteralPath $exportMapFile -Encoding UTF8
            Write-Host ("export_map={0} entries saved to {1}" -f $map.Count, $exportMapFile)
        }
        Remove-Item -LiteralPath $workletTempJs -Force -ErrorAction SilentlyContinue
    }

    $wasmArtifact = Get-Item -LiteralPath $finalWasm
    Write-Host ("worklet_wasm={0} bytes" -f $wasmArtifact.Length)
}
