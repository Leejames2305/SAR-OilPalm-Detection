#!/bin/sh
set -eu
mkdir -p /app/submission /app/report /app/datasets
cp /solution/pipeline.py /app/submission/pipeline.py
cp /solution/requirements.txt /app/submission/requirements.txt
