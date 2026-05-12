#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/package_submission.sh FirstName LastName"
  exit 1
fi

zip_name="${1}_${2}_GalaxEye.zip"
staging="outputs/submission"

mkdir -p "$staging"
cp outputs/checkpoints/best.pth "$staging/best.pth"
cp reports/technical_report.pdf "$staging/technical_report.pdf"
cp reports/time_resource_log.txt "$staging/time_resource_log.txt"
rm -f "$zip_name"
cd "$staging"
zip -r "../../$zip_name" .
echo "Created $zip_name"
