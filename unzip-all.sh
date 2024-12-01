#!/bin/bash

find . -name "*.zip" -type f | while read file; do
   unzip -o "$file" -d "$(dirname "$file")"
done