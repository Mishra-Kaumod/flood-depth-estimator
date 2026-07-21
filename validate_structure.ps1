# FloodWatch AI — Modular Structure Validator
# Run from repo root: .\validate_structure.ps1

$pass = 0; $fail = 0

$checks = @(
    # Root
    @{f='main.py';            desc='Orchestrator (15-min batch loop)'},
    @{f='ingestor.py';        desc='Inbox watcher'},
    @{f='docker-compose.yml'; desc='Docker services'},
    @{f='.env.example';       desc='Config template'},
    @{f='pyproject.toml';     desc='Dependency manifest'},

    # Pipeline
    @{f='pipeline\segformer.py';       desc='Stage 1 - water mask'},
    @{f='pipeline\yolo.py';            desc='Stage 2 - reference objects'},
    @{f='pipeline\depth.py';           desc='Stage 3 - depth map'},
    @{f='pipeline\fusion.py';          desc='Stage 4 - calibrate/merge'},
    @{f='pipeline\severity.py';        desc='Stage 5 - risk prediction'},
    @{f='pipeline\gemini_validator.py';desc='Stage 6 - Gemini ensemble'},
    @{f='pipeline\runner.py';          desc='Pipeline wiring'},
    @{f='pipeline\README.md';          desc='Pipeline docs'},

    # API
    @{f='api\server.py';  desc='FastAPI REST server'},
    @{f='api\README.md';  desc='API docs'},

    # DB
    @{f='db\postgres.py'; desc='PostgreSQL writer'},
    @{f='db\README.md';   desc='DB docs'},

    # UI
    @{f='ui\upload_app.py'; desc='Upload + predict UI'},
    @{f='ui\app.py';        desc='Live dashboard'},
    @{f='ui\README.md';     desc='UI docs'},

    # src infra
    @{f='src\queue\job_queue.py';          desc='Redis queue + DLQ'},
    @{f='src\queue\README.md';             desc='Queue docs'},
    @{f='src\api\middleware.py';           desc='Auth + rate limit'},
    @{f='src\observability\metrics.py';    desc='Prometheus metrics'},
    @{f='src\observability\alerts.py';     desc='Slack/SMS alerts'},
    @{f='src\observability\README.md';     desc='Observability docs'},
    @{f='src\pipeline\worker_pool.py';     desc='N-worker pool'},

    # Config
    @{f='config\config.yaml'; desc='Main config'},
    @{f='config\settings.py'; desc='Pydantic settings'},

    # Tests
    @{f='tests\unit\test_fusion.py';     desc='Fusion unit tests'},
    @{f='tests\unit\test_severity.py';   desc='Severity unit tests'},
    @{f='tests\integration\test_api.py'; desc='API integration tests'},
    @{f='tests\README.md';               desc='Test docs'},

    # Models
    @{f='models\best_floodnet_v2.pth';            desc='Production model (B4)'},
    @{f='models\best_flood_model_water_aware.pth'; desc='Previous model (B0)'},
    @{f='models\README.md';                        desc='Models docs'},

    # Notebooks
    @{f='notebooks\FloodWatch_Upgraded_Architecture.ipynb'; desc='B4 Colab notebook'},
    @{f='notebooks\FloodWatch_MLOps_Training.ipynb';        desc='MLOps notebook'},
    @{f='notebooks\README.md';                              desc='Notebooks docs'}
)

Write-Host "`n======================================================"
Write-Host " FloodWatch AI — Modular Structure Validation"
Write-Host "======================================================`n"

$lastSection = ''
foreach ($c in $checks) {
    $section = ($c.f -split '\\')[0]
    if ($section -ne $lastSection) {
        Write-Host "`n  [$section]"
        $lastSection = $section
    }
    if (Test-Path $c.f) {
        Write-Host "    ✅  $($c.f.PadRight(48)) $($c.desc)"
        $pass++
    } else {
        Write-Host "    ❌  $($c.f.PadRight(48)) MISSING — $($c.desc)"
        $fail++
    }
}

Write-Host "`n======================================================"
Write-Host "  PASSED : $pass / $($pass + $fail)"
Write-Host "  FAILED : $fail"
if ($fail -eq 0) {
    Write-Host "  RESULT : ✅ ALL CHECKS PASSED — repo is fully modular"
} else {
    Write-Host "  RESULT : ❌ $fail file(s) missing — see above"
}
Write-Host "======================================================`n"

exit $fail
