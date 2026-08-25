param(
    [string]$Message = "Update MBLM source",
    [string]$ServerHost = "lab",
    [string]$RemoteRoot = "/home/maisonglang/mblm-cbqa/multiscale-byte-lm",
    [switch]$NoPush,
    [switch]$NoServerDeploy
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Read-Choice {
    param([string]$Prompt)

    $value = Read-Host $Prompt
    if ($null -eq $value) {
        return ""
    }
    return $value.Trim()
}

function Assert-LastExitCode {
    param([string]$Action)

    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE"
    }
}

function Invoke-GitQuiet {
    param([string[]]$GitArgs)

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git @GitArgs 2>$null
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

function Invoke-GitRequired {
    param(
        [string[]]$GitArgs,
        [string]$Action
    )

    $output = & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Git command failed while running: git $($GitArgs -join ' ')"
        Write-Host "If this reports 'detected dubious ownership', run this once in PowerShell:"
        Write-Host "  git config --global --add safe.directory F:/cc_workspace/projects/multiscale-byte-lm"
        throw "$Action failed with exit code $LASTEXITCODE"
    }
    return $output
}

function Test-Command {
    param([string]$Name)

    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-PythonModule {
    param([string]$Module)

    & python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Module') else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

function Invoke-Ruff {
    param([string[]]$RuffArgs)

    if (Test-Command "ruff") {
        & ruff @RuffArgs
        return $LASTEXITCODE
    }

    if (Test-PythonModule "ruff") {
        & python -m ruff @RuffArgs
        return $LASTEXITCODE
    }

    Write-Host "Warning: ruff is not available in PATH or the active Python; skipping ruff checks."
    Write-Host "Checked command paths:"
    Write-Host "  ruff"
    Write-Host "  python -m ruff"
    return 0
}

function Get-ChangedPaths {
    $statusLines = Invoke-GitQuiet @("status", "--porcelain")
    $paths = foreach ($line in $statusLines) {
        if (-not $line -or $line.Length -lt 4) {
            continue
        }
        $rawPath = $line.Substring(3).Trim()
        if ($rawPath.Contains(" -> ")) {
            $rawPath = ($rawPath -split " -> ")[-1].Trim()
        }
        $rawPath.Trim('"')
    }

    $paths | Where-Object { $_ -and $_.Trim() -ne "" } | Sort-Object -Unique
}

function Test-DeployablePath {
    param([string]$Path)

    $p = $Path.Replace("\", "/")
    if ($p.StartsWith("src/")) {
        return $true
    }
    if ($p.StartsWith("scripts/")) {
        return $true
    }
    if ($p.StartsWith("config/")) {
        return $true
    }
    return $p -in @(
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "Makefile",
        ".python-version"
    )
}

function Get-DeployablePaths {
    param([string[]]$Paths)

    $Paths |
        Where-Object { Test-DeployablePath $_ } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        ForEach-Object { $_.Replace("\", "/") } |
        Sort-Object -Unique
}

function Show-ChangeSummary {
    param([string[]]$Paths)

    Write-Host "Change summary:"
    git diff --stat
    git diff --cached --stat

    $untracked = Invoke-GitQuiet @("ls-files", "--others", "--exclude-standard")
    if ($untracked) {
        Write-Host "Untracked files:"
        $untracked | ForEach-Object { Write-Host "  $_" }
    }

    Write-Host ""
    Write-Host "Deploy mapping if server deploy is confirmed:"
    $deployable = @(Get-DeployablePaths -Paths $Paths)
    if (-not $deployable -or $deployable.Count -eq 0) {
        Write-Host "  No deployable changed files found."
        return
    }
    foreach ($path in $deployable) {
        Write-Host "  $path -> $RemoteRoot/$path"
    }
}

function Invoke-Checks {
    $changed = @(Get-ChangedPaths)
    $pythonFiles = @(
        $changed |
            Where-Object { $_.Replace("\", "/").StartsWith("src/") -and $_.EndsWith(".py") } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )

    if ($pythonFiles.Count -eq 0) {
        Write-Host "No changed src/*.py files to check."
        return
    }

    Write-Host "Running py_compile on changed source files..."
    & python -m py_compile @pythonFiles
    Assert-LastExitCode "py_compile"

    Write-Host "Running ruff check on changed source files..."
    $ruffCheckArgs = @("check") + $pythonFiles
    $ruffExitCode = Invoke-Ruff -RuffArgs $ruffCheckArgs
    if ($ruffExitCode -ne 0) {
        throw "ruff check failed with exit code $ruffExitCode"
    }

    Write-Host "Running ruff format --check on changed source files..."
    $ruffFormatArgs = @("format", "--check") + $pythonFiles
    $ruffFormatExitCode = Invoke-Ruff -RuffArgs $ruffFormatArgs
    if ($ruffFormatExitCode -ne 0) {
        throw "ruff format check failed with exit code $ruffFormatExitCode"
    }
    Write-Host ""
}

function Assert-GitIdentity {
    $name = (Invoke-GitQuiet @("config", "--get", "user.name") | Select-Object -First 1)
    $email = (Invoke-GitQuiet @("config", "--get", "user.email") | Select-Object -First 1)

    if ($name -and $email) {
        return
    }

    Write-Host ""
    Write-Host "Git commit identity is not configured for this repository."
    Write-Host "Set it once with your real Git identity, for this repo only:"
    Write-Host "  git config user.name ""Your Name"""
    Write-Host "  git config user.email ""you@example.com"""
    Write-Host ""
    Write-Host "Or use --global if you want it to apply to all local repositories."
    throw "git identity is missing"
}

function Invoke-ServerDeploy {
    param(
        [string]$HostName,
        [string]$RootPath,
        [string[]]$DeployablePaths
    )

    if (-not $DeployablePaths -or $DeployablePaths.Count -eq 0) {
        Write-Host "No deployable changed files; skipping server deploy."
        return
    }

    if (-not (Test-Command "scp")) {
        throw "scp not found in PATH"
    }
    if (-not (Test-Command "ssh")) {
        throw "ssh not found in PATH"
    }
    if (-not (Test-Command "tar")) {
        throw "tar not found in PATH"
    }

    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mblm-deploy-" + [System.Guid]::NewGuid().ToString("N"))
    $stageRoot = Join-Path $tempRoot "payload"
    New-Item -ItemType Directory -Path $stageRoot | Out-Null

    try {
        foreach ($rel in $DeployablePaths) {
            $src = Join-Path $repoRoot $rel
            $dst = Join-Path $stageRoot $rel
            $dstDir = Split-Path -Parent $dst
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            Copy-Item -LiteralPath $src -Destination $dst -Force
        }

        $archive = Join-Path $tempRoot "payload.tar.gz"
        & tar -czf $archive -C $stageRoot .
        Assert-LastExitCode "create deploy archive"

        $remoteTmp = "/tmp/mblm-deploy-$([System.Guid]::NewGuid().ToString("N")).tar.gz"
        & scp $archive "${HostName}:$remoteTmp"
        Assert-LastExitCode "scp deploy archive"

        $remoteScript = @"
set -euo pipefail
REMOTE_ROOT="$RootPath"
REMOTE_TMP="$remoteTmp"
case "`$REMOTE_ROOT" in
  /home/maisonglang/mblm-cbqa/multiscale-byte-lm) ;;
  *) echo "Refusing unsafe remote root: `$REMOTE_ROOT" >&2; exit 2 ;;
esac
if [[ ! -d "`$REMOTE_ROOT" ]]; then
  echo "Remote MBLM source directory does not exist: `$REMOTE_ROOT" >&2
  exit 3
fi
tar -xzf "`$REMOTE_TMP" -C "`$REMOTE_ROOT"
rm -f "`$REMOTE_TMP"
echo "Deployed files into `$REMOTE_ROOT"
"@
        $remoteScript = $remoteScript -replace "`r`n", "`n"
        $remoteScript = $remoteScript -replace "`r", "`n"

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "ssh"
        $psi.Arguments = "$HostName ""bash -s"""
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $process = [System.Diagnostics.Process]::Start($psi)
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        $bytes = $utf8NoBom.GetBytes($remoteScript)
        $process.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
        $process.StandardInput.Close()
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) {
            throw "server deploy failed with exit code $($process.ExitCode)"
        }
    } finally {
        if (Test-Path -LiteralPath $tempRoot) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "Repository: $repoRoot"
$head = (Invoke-GitRequired @("rev-parse", "HEAD") "git rev-parse HEAD" | Select-Object -First 1)
Write-Host "HEAD: $($head.Trim())"
Write-Host "Origin:"
Invoke-GitRequired @("remote", "-v") "git remote -v"
Write-Host ""

Invoke-Checks
Assert-GitIdentity

$status = git status --short
if (-not $status) {
    Write-Host "No local changes."
    exit 0
}

Write-Host "Detected changes:"
$status | ForEach-Object { Write-Host "  $_" }
Write-Host ""

$changedPaths = @(Get-ChangedPaths)
Show-ChangeSummary -Paths $changedPaths
Write-Host ""
Write-Host "Commit message: $Message"

while ($true) {
    $answer = Read-Choice "Commit changes? Enter y to continue, d for diff, anything else to cancel"
    if ($answer -eq "d" -or $answer -eq "D") {
        git diff -- $changedPaths
        Write-Host ""
        continue
    }
    if ($answer -eq "y" -or $answer -eq "Y") {
        break
    }
    Write-Host "Cancelled; no add, commit, push, or deploy was run."
    exit 0
}

git add .
Assert-LastExitCode "git add"
git commit -m $Message
Assert-LastExitCode "git commit"

if (-not $NoPush) {
    $pushUrl = (git remote get-url --push origin).Trim()
    if ($pushUrl -eq "https://github.com/ai4sd/multiscale-byte-lm.git") {
        Write-Host ""
        Write-Host "Origin push URL is the upstream official repo:"
        Write-Host "  $pushUrl"
        Write-Host "Skipping git push. Configure origin to your fork, or rerun with -NoPush intentionally."
    } else {
        git push
        Assert-LastExitCode "git push"
    }
}

if (-not $NoServerDeploy) {
    Write-Host ""
    $headFiles = @(git diff-tree --no-commit-id --name-only -r HEAD)
    $deployable = @(Get-DeployablePaths -Paths $headFiles)
    Write-Host "Server deploy will copy these files:"
    $deployable | ForEach-Object { Write-Host "  $_ -> $RemoteRoot/$_" }
    $deployAnswer = Read-Choice "Deploy changed source files to server? Enter y to continue"
    if ($deployAnswer -eq "y" -or $deployAnswer -eq "Y") {
        Invoke-ServerDeploy -HostName $ServerHost -RootPath $RemoteRoot -DeployablePaths $deployable
    } else {
        Write-Host "Skipped server deploy."
    }
}

Write-Host ""
Write-Host "Done."
