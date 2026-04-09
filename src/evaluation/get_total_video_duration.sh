#!/bin/bash

videoFolder="./dataset/raw/"
totalSeconds=0

find "$videoFolder" -type f -name "*.mp4" | while read -r file; do
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")

    totalSeconds=$(echo "$totalSeconds + $duration" | bc)

    hours=$(echo "$duration / 3600" | bc)
    minutes=$(echo "($duration % 3600) / 60" | bc)
    seconds=$(echo "$duration % 60" | bc)
    printf "[File] %s: %02d:%02d:%02d\n" "$(basename "$file")" "$hours" "$minutes" "$seconds"
done

totalHours=$(echo "$totalSeconds / 3600" | bc)
totalMinutes=$(echo "($totalSeconds % 3600) / 60" | bc)
totalSecondsOnly=$(echo "$totalSeconds % 60" | bc)

echo ""
echo "--------------------------------"
printf "Total Duration: %02d:%02d:%02d\n" "$totalHours" "$totalMinutes" "$totalSecondsOnly"