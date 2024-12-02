#!/bin/bash

# 7zip이 설치되어 있는지 확인
if ! command -v 7z &> /dev/null; then
    echo "7zip이 설치되어 있지 않습니다. 설치해주세요."
    echo "Ubuntu: sudo apt-get install p7zip-full"
    echo "CentOS: sudo yum install p7zip p7zip-plugins"
    echo "macOS: brew install p7zip"
    exit 1
fi

find . -type f -name "*.zip" -print0 | while IFS= read -r -d '' file; do
   echo "압축 해제 중: $file"
   dirname=$(dirname "$file")
   filename=$(basename "$file" .zip)
   target_dir="$dirname"
   mkdir -p "$target_dir"
   7z x -o"$target_dir" "$file"
done