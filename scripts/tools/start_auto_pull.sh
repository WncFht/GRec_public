#!/bin/bash

# 启动自动拉取代码脚本
# 用于在后台启动和管理 auto_pull.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_PULL_SCRIPT="$SCRIPT_DIR/auto_pull.sh"
PID_FILE="$SCRIPT_DIR/auto_pull.pid"
LOG_FILE="$SCRIPT_DIR/auto_pull.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 检查脚本是否存在
check_script() {
    if [ ! -f "$AUTO_PULL_SCRIPT" ]; then
        print_message $RED "错误: 找不到脚本 $AUTO_PULL_SCRIPT"
        exit 1
    fi
    
    # 给脚本添加执行权限
    chmod +x "$AUTO_PULL_SCRIPT"
}

# 检查是否在git仓库中
check_git_repo() {
    if [ ! -d ".git" ]; then
        print_message $RED "错误: 当前目录不是git仓库"
        exit 1
    fi
}

# 启动脚本
start_script() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_message $YELLOW "脚本已经在运行中，PID: $(cat "$PID_FILE")"
        return
    fi
    
    print_message $GREEN "正在启动自动拉取代码脚本..."
    
    # 在后台启动脚本
    nohup "$AUTO_PULL_SCRIPT" > /dev/null 2>&1 &
    local pid=$!
    
    # 保存PID到文件
    echo $pid > "$PID_FILE"
    
    # 等待一下确认启动成功
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        print_message $GREEN "脚本启动成功！PID: $pid"
        print_message $GREEN "日志文件: $LOG_FILE"
        print_message $GREEN "停止脚本: $0 stop"
    else
        print_message $RED "脚本启动失败"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# 停止脚本
stop_script() {
    if [ ! -f "$PID_FILE" ]; then
        print_message $YELLOW "没有找到PID文件，脚本可能未运行"
        return
    fi
    
    local pid=$(cat "$PID_FILE")
    if kill -0 $pid 2>/dev/null; then
        print_message $YELLOW "正在停止脚本 (PID: $pid)..."
        kill $pid
        
        # 等待进程结束
        local count=0
        while kill -0 $pid 2>/dev/null && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # 如果还在运行，强制终止
        if kill -0 $pid 2>/dev/null; then
            print_message $YELLOW "强制终止进程..."
            kill -9 $pid
        fi
        
        print_message $GREEN "脚本已停止"
    else
        print_message $YELLOW "进程 $pid 不存在"
    fi
    
    # 清理PID文件
    rm -f "$PID_FILE"
}

# 查看状态
show_status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 $pid 2>/dev/null; then
            print_message $GREEN "脚本正在运行，PID: $pid"
            print_message $GREEN "日志文件: $LOG_FILE"
            echo
            print_message $YELLOW "最后10行日志:"
            if [ -f "$LOG_FILE" ]; then
                tail -10 "$LOG_FILE"
            else
                print_message $YELLOW "日志文件不存在"
            fi
        else
            print_message $RED "PID文件存在但进程已死亡，清理PID文件"
            rm -f "$PID_FILE"
        fi
    else
        print_message $YELLOW "脚本未运行"
    fi
}

# 重启脚本
restart_script() {
    print_message $YELLOW "重启脚本..."
    stop_script
    sleep 2
    start_script
}

# 主函数
main() {
    case "${1:-start}" in
        start)
            check_script
            check_git_repo
            start_script
            ;;
        stop)
            stop_script
            ;;
        restart)
            check_script
            check_git_repo
            restart_script
            ;;
        status)
            show_status
            ;;
        *)
            echo "用法: $0 {start|stop|restart|status}"
            echo "  start   - 启动脚本"
            echo "  stop    - 停止脚本"
            echo "  restart - 重启脚本"
            echo "  status  - 查看状态"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
