$videoFolder = "./dataset/raw/"
$totalSeconds = 0
$videoFiles = Get-ChildItem -Path $videoFolder -Filter *.mp4 -Recurse
foreach ($file in $videoFiles) {
    $duration = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file.FullName
    $totalSeconds += [float]$duration
    $timeSpan = [TimeSpan]::FromSeconds($duration)
    Write-Host "[File] $($file.Name): $($timeSpan.ToString('hh\:mm\:ss'))"
}
$totalTime = [TimeSpan]::FromSeconds($totalSeconds)
Write-Host "`n--------------------------------"
Write-Host "Total Duration: $($totalTime.ToString('hh\:mm\:ss'))"