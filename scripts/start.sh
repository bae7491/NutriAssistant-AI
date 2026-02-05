#!/bin/bash

# ===========================================
# FastAPI 서버 시작 스크립트
# 사용법: ./start.sh [local|aws]
# 인자가 없으면 .env의 APP_ENV 값을 사용
# ===========================================

# 스크립트 위치 기준으로 프로젝트 루트 찾기
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# .env 파일 로드
if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep -v '^$' | xargs)
fi

# 환경 설정 (인자가 있으면 인자 사용, 없으면 .env의 APP_ENV 사용)
ENV=${1:-${APP_ENV:-aws}}

echo "========================================"
echo "FastAPI Server Starting..."
echo "Environment: $ENV"
echo "========================================"

if [ "$ENV" = "aws" ]; then
  # ===========================================
  # AWS 환경 설정
  # ===========================================
  HOST=${AWS_HOST:-0.0.0.0}
  PORT=${AWS_PORT:-8001}
  APP_DIR=${AWS_APP_DIR:-/home/ec2-user/ai-app}
  LOG_FILE="$APP_DIR/ai-app.log"

  echo "Host: $HOST"
  echo "Port: $PORT"
  echo "App Directory: $APP_DIR"
  echo "Log File: $LOG_FILE"
  echo "========================================"

  cd "$APP_DIR"
  nohup python3.11 -m uvicorn app.main:app --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &

  echo "Server started in background. Check logs: tail -f $LOG_FILE"

elif [ "$ENV" = "local" ]; then
  # ===========================================
  # 로컬 환경 설정
  # ===========================================
  HOST=${LOCAL_HOST:-127.0.0.1}
  PORT=${LOCAL_PORT:-8000}

  echo "Host: $HOST"
  echo "Port: $PORT"
  echo "App Directory: $PROJECT_ROOT"
  echo "========================================"

  cd "$PROJECT_ROOT"

  # 가상환경 활성화 (있는 경우)
  if [ -d ".venv" ]; then
      source .venv/bin/activate
      echo "Virtual environment activated"
  fi

  # 개발 모드로 실행 (--reload 옵션)
  python -m uvicorn app.main:app --reload --host "$HOST" --port "$PORT"

else
  echo "ERROR: Unknown environment '$ENV'"
  echo "Usage: ./start.sh [local|aws]"
  echo "Or set APP_ENV in .env file"
  exit 1
fi
