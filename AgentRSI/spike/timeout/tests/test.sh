#!/bin/sh
mkdir -p /logs/verifier
if [ "$(cat /app/out.txt 2>/dev/null)" = "hello from agentrsi" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
