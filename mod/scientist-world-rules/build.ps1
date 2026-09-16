[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $GradleArguments
)

$ErrorActionPreference = "Stop"

if (-not $GradleArguments -or $GradleArguments.Count -eq 0) {
    $GradleArguments = @("clean", "build")
}

function Test-JavaHome([string] $Path) {
    return $Path -and (Test-Path -LiteralPath (Join-Path $Path "bin\java.exe"))
}

if (-not (Test-JavaHome $env:JAVA_HOME)) {
    $roots = @(
        "C:\Program Files\Eclipse Adoptium",
        "C:\Program Files\Microsoft"
    )
    $candidate = $roots |
        Where-Object { Test-Path -LiteralPath $_ } |
        ForEach-Object { Get-ChildItem -LiteralPath $_ -Directory -ErrorAction SilentlyContinue } |
        Where-Object { Test-JavaHome $_.FullName } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "Java 21 was not found. Set JAVA_HOME to a JDK 21 installation."
    }
    $env:JAVA_HOME = $candidate.FullName
}

$javaTmp = Join-Path $PSScriptRoot ".java-tmp"
New-Item -ItemType Directory -Force -Path $javaTmp | Out-Null
$env:TEMP = $javaTmp
$env:TMP = $javaTmp

$compatOptions = @(
    "-Djava.io.tmpdir=$javaTmp",
    "-Djdk.net.unixdomain.tmpdir=$javaTmp",
    "-Djava.net.preferIPv4Stack=true"
) -join " "
$env:JAVA_TOOL_OPTIONS = (@($env:JAVA_TOOL_OPTIONS, $compatOptions) |
    Where-Object { $_ }) -join " "

& (Join-Path $PSScriptRoot "gradlew.bat") --no-daemon --project-dir $PSScriptRoot @GradleArguments
if ($LASTEXITCODE -ne 0) {
    throw "Gradle failed with exit code $LASTEXITCODE."
}
