#!/bin/bash
# LLM 训练守护进程脚本
# 使用 nohup 在后台运行，终端关闭后继续训练
#
# 用法:
#   ./scripts/train_daemon.sh start    # 启动后台训练
#   ./scripts/train_daemon.sh stop     # 停止训练
#   ./scripts/train_daemon.sh status   # 查看状态
#   ./scripts/train_daemon.sh log      # 实时查看日志
#   ./scripts/train_daemon.sh restart  # 重启训练

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.train_pid"
LOG_FILE="$PROJECT_DIR/logs/train_llm.log"
ERR_FILE="$PROJECT_DIR/logs/train_llm_err.log"
PYTHON="${PYTHON:-python3}"

# 自动检测虚拟环境
if [ -z "$PYTHON_SET" ]; then
    if [ -f "$PROJECT_DIR/../.venv/bin/python" ]; then
        PYTHON="$PROJECT_DIR/../.venv/bin/python"
    elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
        PYTHON="$PROJECT_DIR/.venv/bin/python"
    fi
fi

mkdir -p "$PROJECT_DIR/logs"

case "$1" in
    start)
        # 检查是否已在运行
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "训练已在运行中 (PID: $PID)"
                echo "查看日志: ./scripts/train_daemon.sh log"
                exit 1
            fi
        fi

        # 环境变量
        export HF_ENDPOINT="https://hf-mirror.com"
        export HF_HUB_OFFLINE="0"
        export TRANSFORMERS_OFFLINE="0"

        echo "启动后台训练..."
        echo "Python: $PYTHON"
        echo "日志文件: $LOG_FILE"
        echo "错误文件: $ERR_FILE"

        cd "$PROJECT_DIR"
        # 如果有已保存的 checkpoint, 自动恢复训练
        if [ -f "$PROJECT_DIR/checkpoints/llm_mps/checkpoint-interrupted" ] || \
           [ -d "$PROJECT_DIR/checkpoints/llm_mps" ]; then
            echo "检测到已有 checkpoint，将自动恢复训练..."
            nohup $PYTHON scripts/train_llm.py --auto --resume auto > "$LOG_FILE" 2> "$ERR_FILE" &
        else
            nohup $PYTHON scripts/train_llm.py --auto > "$LOG_FILE" 2> "$ERR_FILE" &
        fi
        PID=$!
        echo "$PID" > "$PID_FILE"

        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "训练已启动 (PID: $PID)"
            echo "查看日志: ./scripts/train_daemon.sh log"
            echo "查看状态: ./scripts/train_daemon.sh status"
            echo "停止训练: ./scripts/train_daemon.sh stop"
        else
            echo "启动失败! 查看错误日志: $ERR_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                kill "$PID"
                sleep 2
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID"
                fi
                echo "训练已停止 (PID: $PID)"
            else
                echo "进程不存在 (PID: $PID)"
            fi
            rm -f "$PID_FILE"
        else
            echo "未找到 PID 文件，训练未在运行"
        fi
        ;;

    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "训练运行中 (PID: $PID)"
                # 显示最后几行日志
                echo ""
                echo "--- 最新进度 ---"
                tail -3 "$LOG_FILE" 2>/dev/null
            else
                echo "进程已退出 (PID: $PID)"
                echo "查看日志: ./scripts/train_daemon.sh log"
            fi
        else
            echo "训练未在运行"
        fi
        ;;

    log)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "日志文件不存在: $LOG_FILE"
        fi
        ;;

    err)
        if [ -f "$ERR_FILE" ]; then
            tail -f "$ERR_FILE"
        else
            echo "错误日志不存在: $ERR_FILE"
        fi
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start
        ;;

    *)
        echo "用法: $0 {start|stop|status|log|err|restart}"
        echo ""
        echo "命令说明:"
        echo "  start    启动后台训练"
        echo "  stop     停止训练"
        echo "  status   查看训练状态"
        echo "  log      实时查看训练日志"
        echo "  err      实时查看错误日志"
        echo "  restart  重启训练"
        exit 1
        ;;
esac
