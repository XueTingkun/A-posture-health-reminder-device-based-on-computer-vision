$targetDir = "./dataset/raw"

$files = Get-ChildItem -Path $targetDir -Filter *.mp4

foreach ($file in $files) {
    Write-Host "Processing: $($file.Name)"
    $tempPath = "$($file.FullName).tmp.mp4"

    if (Test-Path $tempPath) {
        Remove-Item -Path $tempPath -Force
    }

    $process = Start-Process -FilePath "ffmpeg" -ArgumentList "-i `"$($file.FullName)`" -an -c:v copy -y `"$tempPath`"" -NoNewWindow -Wait -PassThru

    if ($process.ExitCode -eq 0) {
        Write-Host "Success: $($file.Name)" -ForegroundColor Green
        Remove-Item -Path $file.FullName -Force
        Move-Item -Path $tempPath -Destination $file.FullName
    } else {
        Write-Host "Failed: $($file.Name)" -ForegroundColor Red
        if (Test-Path $tempPath) {
            Remove-Item -Path $tempPath -Force
        }
    }
}
