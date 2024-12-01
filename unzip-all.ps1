$zipFiles = Get-ChildItem -Path . -Filter *.zip -Recurse

foreach ($zip in $zipFiles) {
    Expand-Archive -Path $zip.FullName -DestinationPath $zip.DirectoryName -Force
}