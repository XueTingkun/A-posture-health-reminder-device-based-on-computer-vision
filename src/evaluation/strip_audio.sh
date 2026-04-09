#!/bin/bash

targetDir="./dataset/raw"
cd "$targetDir" || exit 1

for file in *.mp4; do
    [ -e "$file" ] || continue

    echo "Processing: $file"
    tempPath="${file}.tmp.mp4"

    [ -f "$tempPath" ] && rm -f "$tempPath"

    ffmpeg -i "$file" -an -c:v copy -y "$tempPath"

    if [ $? -eq 0 ]; then
        echo -e "\e[32mSuccess: $file\e[0m"
        rm -f "$file"
        mv "$tempPath" "$file"
    else
        echo -e "\e[31mFailed: $file\e[0m"
        [ -f "$tempPath" ] && rm -f "$tempPath"
    fi
donef