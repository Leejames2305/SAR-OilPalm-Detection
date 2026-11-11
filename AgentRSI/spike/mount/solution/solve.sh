#!/bin/sh
echo "hello from agentrsi" > /app/out.txt
if [ -r /run/secrets/gcp-sa.json ]; then echo "readable" > /app/mount-status.txt; else echo "missing" > /app/mount-status.txt; fi
if (echo x >> /run/secrets/gcp-sa.json) 2>/dev/null; then echo "WRITABLE" >> /app/mount-status.txt; else echo "readonly-ok" >> /app/mount-status.txt; fi
