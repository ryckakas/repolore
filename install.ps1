# Installs repolore with uv, then runs `repolore setup` with any arguments given:
#
#   irm https://raw.githubusercontent.com/ryckakas/repolore/main/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/ryckakas/repolore/main/install.ps1))) --register
#
# $env:REPOLORE_INSTALL_FROM_GIT = '1' installs the latest main instead of the last release.
#
# Failures throw and never exit: under iex, exit would close the terminal this runs in. The script block keeps its
# functions and variables out of that terminal's session.

& {
    # A caller's Stop would turn uv's progress on stderr into errors on Windows PowerShell.
    $ErrorActionPreference = 'Continue'
    $package = 'repolore-mcp'
    $gitSource = 'git+https://github.com/ryckakas/repolore'
    $uvDocs = 'https://docs.astral.sh/uv/getting-started/installation/'

    function Test-Command([string]$Name) {
        [bool](Get-Command $Name -ErrorAction SilentlyContinue)
    }

    function Get-UvFolder {
        if ($env:XDG_BIN_HOME) { return $env:XDG_BIN_HOME }
        if ($env:XDG_DATA_HOME) { return Join-Path $env:XDG_DATA_HOME '..\bin' }
        Join-Path $env:USERPROFILE '.local\bin'
    }

    function Install-Uv {
        if (Test-Command 'uv') { return }
        Write-Host 'repolore: installing uv'
        # In a child, as uv documents: its installer exits on failure, which would close this terminal under iex.
        powershell -NoProfile -ExecutionPolicy ByPass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
        $env:Path = "$(Get-UvFolder);$env:Path"
        if (-not (Test-Command 'uv')) { throw "repolore: couldn't install uv; see $uvDocs" }
    }

    function Test-Installed {
        $listed = @(uv tool list 2>$null)
        if ($LASTEXITCODE -ne 0) { throw "repolore: couldn't list uv's tools" }
        [bool]($listed -match "^$package ")
    }

    function Install-Repolore {
        # Windows keeps a running program's files locked, so the shared server would block the upgrade.
        if (Test-Command 'repolore') { repolore server stop }
        if ($env:REPOLORE_INSTALL_FROM_GIT -eq '1') {
            Write-Host 'repolore: installing the latest main; extras such as rerank have to be added again'
            uv tool install --managed-python --python 3.12 --upgrade $gitSource
        } elseif (Test-Installed) {
            uv tool upgrade $package
        } else {
            uv tool install --managed-python --python 3.12 $package
        }
        if ($LASTEXITCODE -ne 0) {
            throw "repolore: couldn't install $package; close the sessions using repolore and try again"
        }
    }

    function Add-ToolsToPath {
        $tools = uv tool dir --bin
        if ($LASTEXITCODE -ne 0) { throw "repolore: couldn't find uv's tool folder" }
        if (($env:Path -split ';') -notcontains $tools) {
            # It errors when the shell files have the folder but this process doesn't, which is the case right after
            # a first install.
            uv tool update-shell *> $null
            $env:Path = "$tools;$env:Path"
            Write-Host 'repolore: open a new terminal to use the repolore command'
        }
    }

    Install-Uv
    Install-Repolore
    Add-ToolsToPath
    if (-not (Test-Command 'repolore')) { throw "repolore: the repolore command wasn't found after installing it" }

    # An unquoted a,b arrives as an array, which would reach setup as two arguments.
    $setupArgs = @($args | ForEach-Object { $_ -join ',' })
    if ([Console]::IsInputRedirected -and $setupArgs.Count -eq 0) {
        Write-Host "repolore: installed; there's no terminal to ask questions in, so run ``repolore setup`` in one to finish"
        return
    }
    repolore setup @setupArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'repolore: setup stopped; its output says why, and without a terminal every question needs its flag'
    }
} @args
