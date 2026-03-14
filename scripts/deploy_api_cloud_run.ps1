param(
    [string]$ProjectId = "",
    [string]$Region = "us-central1",
    [string]$ServiceName = "college-basketball-dfs-api",
    [Parameter(Mandatory = $true)]
    [string]$BucketName,
    [string]$ServiceAccountJsonPath = "",
    [string]$SecretName = "college-basketball-dfs-api-service-account-json"
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectId {
    param([string]$ConfiguredProjectId)

    if ($ConfiguredProjectId) {
        return $ConfiguredProjectId.Trim()
    }

    $resolved = (& gcloud config get-value project 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the active gcloud project. Pass -ProjectId explicitly."
    }
    $resolved = ($resolved | Out-String).Trim()
    if (-not $resolved -or $resolved -eq "(unset)") {
        throw "No active gcloud project is set. Run 'gcloud config set project <PROJECT_ID>' or pass -ProjectId."
    }
    return $resolved
}

function Ensure-SecretVersion {
    param(
        [string]$ResolvedProjectId,
        [string]$ResolvedSecretName,
        [string]$JsonPath
    )

    if (-not (Test-Path -LiteralPath $JsonPath)) {
        throw "Service account JSON file not found: $JsonPath"
    }

    & gcloud secrets describe $ResolvedSecretName --project $ResolvedProjectId *> $null
    if ($LASTEXITCODE -ne 0) {
        & gcloud secrets create $ResolvedSecretName --replication-policy=automatic --project $ResolvedProjectId
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Secret Manager secret '$ResolvedSecretName'."
        }
    }

    & gcloud secrets versions add $ResolvedSecretName --data-file=$JsonPath --project $ResolvedProjectId
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload a secret version from $JsonPath."
    }
}

$ResolvedProjectId = Resolve-ProjectId -ConfiguredProjectId $ProjectId
$ResolvedBucketName = $BucketName.Trim()
if (-not $ResolvedBucketName) {
    throw "BucketName is required."
}

Write-Host "Project: $ResolvedProjectId"
Write-Host "Region: $Region"
Write-Host "Service: $ServiceName"
Write-Host "Bucket:  $ResolvedBucketName"

& gcloud config set project $ResolvedProjectId | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to set gcloud project to $ResolvedProjectId."
}

$deployArgs = @(
    "run",
    "deploy",
    $ServiceName,
    "--source",
    ".",
    "--project",
    $ResolvedProjectId,
    "--region",
    $Region,
    "--platform",
    "managed",
    "--allow-unauthenticated",
    "--port",
    "8080",
    "--cpu",
    "1",
    "--memory",
    "2Gi",
    "--timeout",
    "900",
    "--set-env-vars",
    "CBB_GCS_BUCKET=$ResolvedBucketName,GCP_PROJECT=$ResolvedProjectId"
)

if ($ServiceAccountJsonPath.Trim()) {
    $ResolvedSecretName = $SecretName.Trim()
    if (-not $ResolvedSecretName) {
        throw "SecretName cannot be blank when ServiceAccountJsonPath is supplied."
    }
    Ensure-SecretVersion -ResolvedProjectId $ResolvedProjectId -ResolvedSecretName $ResolvedSecretName -JsonPath $ServiceAccountJsonPath
    $deployArgs += @("--set-secrets", "GCP_SERVICE_ACCOUNT_JSON=$ResolvedSecretName:latest")
    Write-Host "Secret Manager: $ResolvedSecretName"
} else {
    Write-Warning "No service account JSON path supplied. Cloud Run will rely on ADC/workload identity."
}

Write-Host ""
Write-Host "Deploying Cloud Run service..."
& gcloud @deployArgs
if ($LASTEXITCODE -ne 0) {
    throw "Cloud Run deploy failed."
}

$serviceUrl = (& gcloud run services describe $ServiceName --project $ResolvedProjectId --region $Region --format="value(status.url)")
if ($LASTEXITCODE -ne 0) {
    throw "Cloud Run deploy succeeded, but the service URL could not be read."
}
$serviceUrl = ($serviceUrl | Out-String).Trim()
if (-not $serviceUrl) {
    throw "Cloud Run deploy succeeded, but the service URL was empty."
}

Write-Host ""
Write-Host "Service URL: $serviceUrl"
Write-Host "Health URL:  $serviceUrl/health"
Write-Host "Status URL:  $serviceUrl/v1/slates/2026-03-14/main/status?contest_type=Classic&slate_name=All"

Write-Host ""
Write-Host "Running smoke checks..."

try {
    $healthPayload = Invoke-RestMethod -Uri "$serviceUrl/health" -Method Get -TimeoutSec 60
    Write-Host "Health OK: $($healthPayload.ok)"
} catch {
    Write-Warning "Health check failed: $($_.Exception.Message)"
}

try {
    $statusPayload = Invoke-RestMethod -Uri "$serviceUrl/v1/slates/2026-03-14/main/status?contest_type=Classic&slate_name=All" -Method Get -TimeoutSec 120
    Write-Host "Slate status OK: active_source=$($statusPayload.active_source.label) ready=$($statusPayload.active_source.ready)"
} catch {
    Write-Warning "Slate status check failed: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Next step:"
Write-Host "Set NEXT_PUBLIC_API_BASE_URL=$serviceUrl in your Vercel project and redeploy."
