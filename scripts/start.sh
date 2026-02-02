#!/bin/bash
cd /home/ec2-user/ai-app
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /home/ec2-user/ai-app.log 2>&1 &
