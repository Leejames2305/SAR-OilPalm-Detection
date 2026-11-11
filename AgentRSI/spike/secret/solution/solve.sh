#!/bin/sh
echo "hello from agentrsi" > /app/out.txt
# Record only whether the variable arrived and its length. Never write the value.
if [ -n "$M0_SECRET_MARKER" ]; then echo "present len=${#M0_SECRET_MARKER}" > /app/env-status.txt; else echo "absent" > /app/env-status.txt; fi
