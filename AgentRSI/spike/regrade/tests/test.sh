#!/bin/sh
# Edited verifier for the regrade test: awards 0.5 for the same correct output.
mkdir -p /logs/verifier
if [ "$(cat /app/out.txt 2>/dev/null)" = "hello from agentrsi" ]; then
  echo 0.5 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
