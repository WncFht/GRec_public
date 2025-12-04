#!/bin/bash
# enhanced_gpu_manager.sh

PYTHON_SCRIPT="enhanced_gpu_stress.py"
SCRIPT_PID_FILE="./tmp/gpu_stress.pid"
LOG_DIR="./tmp/gpu_stress_logs"
LOG_FILE="$LOG_DIR/gpu_manager.log"
PYTHON_LOG_FILE="$LOG_DIR/gpu_stress_python.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE"
}

# 显示帮助信息
show_help() {
    echo "增强版GPU管理工具"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start [GPUS]         启动GPU压力测试"
    echo "    GPUS可以是:"
    echo "      all              - 使用所有GPU"
    echo "      remaining        - 使用剩余GPU"
    echo "      0,1,2            - 指定GPU列表"
    echo "      0-3              - GPU范围"
    echo "      0                - 单个GPU"
    echo ""
    echo "  stop                 停止所有GPU压力测试"
    echo "  status               查看当前状态"
    echo "  list                 列出可用GPU"
    echo "  monitor              监控GPU使用情况"
    echo "  kill                 强制杀死所有相关进程"
    echo "  logs                 查看日志"
    echo "  clear-logs           清除日志"
    echo ""
    echo "额外参数:"
    echo "  --duration SECONDS   运行时长"
    echo "  --memory-ratio RATIO 内存占用比例 (0.0-1.0)"
    echo ""
    echo "示例:"
    echo "  $0 start all"
    echo "  $0 start 0,1,2 --duration 300"
    echo "  $0 start remaining --memory-ratio 0.8"
    echo "  $0 start 0-2"
    echo "  $0 stop"
    echo ""
    echo "日志文件位置: $LOG_DIR"
}

# 检查依赖
check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3"
        return 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未找到 nvidia-smi"
        return 1
    fi
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "未找到 $PYTHON_SCRIPT"
        return 1
    fi
    
    return 0
}

# 列出GPU信息
list_gpus() {
    log_info "列出GPU信息"
    echo "=== GPU信息 ==="
    python3 "$PYTHON_SCRIPT" --list-gpus 2>&1 | tee -a "$LOG_FILE"
}

# 启动GPU压力测试
start_gpus() {
    local gpu_spec="$1"
    shift
    local extra_args="$@"
    
    if [ -z "$gpu_spec" ]; then
        log_error "请指定GPU"
        echo "错误: 请指定GPU"
        return 1
    fi
    
    log_info "尝试启动GPU压力测试: GPU=$gpu_spec, 参数=$extra_args"
    
    # 检查是否已有进程在运行
    if [ -f "$SCRIPT_PID_FILE" ]; then
        local old_pid=$(cat "$SCRIPT_PID_FILE")
        if ps -p "$old_pid" > /dev/null 2>&1; then
            log_warn "检测到已有进程在运行 (PID: $old_pid)"
            echo "警告: 检测到已有进程在运行 (PID: $old_pid)"
            echo "请先使用 'stop' 命令停止，或使用 'kill' 强制杀死"
            return 1
        else
            log_info "清理过期的PID文件"
            rm -f "$SCRIPT_PID_FILE"
        fi
    fi
    
    echo "启动GPU压力测试: $gpu_spec"
    echo "额外参数: $extra_args"
    echo "日志文件: $PYTHON_LOG_FILE"
    
    # 启动进程，将Python输出重定向到日志文件
    python3 "$PYTHON_SCRIPT" "$gpu_spec" $extra_args >> "$PYTHON_LOG_FILE" 2>&1 &
    local pid=$!
    
    # 保存PID
    echo $pid > "$SCRIPT_PID_FILE"
    
    log_info "GPU压力测试已启动，PID: $pid"
    
    # 等待几秒检查是否成功启动
    sleep 3
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "GPU压力测试已启动，PID: $pid"
        echo "使用 'status' 查看状态，'stop' 停止测试"
        echo "查看日志: $0 logs"
        log_info "GPU压力测试启动成功"
    else
        echo "警告: 进程可能启动失败，请查看日志"
        log_warn "GPU压力测试可能启动失败"
        rm -f "$SCRIPT_PID_FILE"
    fi
}

# 停止GPU压力测试
stop_gpus() {
    log_info "尝试停止GPU压力测试"
    
    if [ ! -f "$SCRIPT_PID_FILE" ]; then
        log_warn "没有检测到运行中的GPU压力测试进程"
        echo "没有检测到运行中的GPU压力测试进程"
        # 作为后备，杀死所有相关进程
        pkill -f "$PYTHON_SCRIPT" 2>/dev/null
        return 0
    fi
    
    local pid=$(cat "$SCRIPT_PID_FILE")
    echo "正在停止GPU压力测试进程 (PID: $pid)..."
    log_info "正在停止GPU压力测试进程 (PID: $pid)"
    
    # 发送终止信号
    if kill -TERM "$pid" 2>/dev/null; then
        # 等待进程结束
        local count=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "进程未正常退出，强制杀死..."
            log_warn "进程未正常退出，强制杀死 (PID: $pid)"
            kill -KILL "$pid" 2>/dev/null
        else
            log_info "进程正常退出"
        fi
    else
        log_warn "无法发送终止信号到进程 $pid"
    fi
    
    rm -f "$SCRIPT_PID_FILE"
    echo "GPU压力测试已停止"
    log_info "GPU压力测试已停止"
}

# 强制杀死所有相关进程
kill_all() {
    log_info "强制杀死所有GPU压力测试进程"
    echo "强制杀死所有GPU压力测试进程..."
    
    pkill -f "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
    sleep 1
    pkill -9 -f "$PYTHON_SCRIPT" 2>/dev/null >> "$LOG_FILE" 2>&1
    
    rm -f "$SCRIPT_PID_FILE"
    echo "所有相关进程已杀死"
    log_info "所有相关进程已杀死"
}

# 查看状态
show_status() {
    log_info "查看GPU压力测试状态"
    echo "=== GPU压力测试状态 ==="
    
    if [ -f "$SCRIPT_PID_FILE" ]; then
        local pid=$(cat "$SCRIPT_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "状态: 运行中"
            echo "PID: $pid"
            echo "启动时间: $(ps -o lstart= -p $pid 2>/dev/null)"
            echo "日志文件: $PYTHON_LOG_FILE"
            echo ""
            echo "进程详情:"
            ps -o pid,ppid,cmd,etime -p "$pid"
            log_info "状态: 运行中 (PID: $pid)"
        else
            echo "状态: 已停止 (PID文件存在但进程不存在)"
            log_info "状态: 已停止 (PID文件存在但进程不存在)"
            rm -f "$SCRIPT_PID_FILE"
        fi
    else
        echo "状态: 未运行"
        log_info "状态: 未运行"
    fi
    
    echo ""
    echo "=== 当前GPU使用情况 ==="
    log_info "获取当前GPU使用情况"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r id name util mem_used mem_total; do
        mem_percent=$(awk "BEGIN {printf \"%.1f\", $mem_used/$mem_total*100}")
        echo "GPU $id: $name - GPU利用率: ${util}% - 内存: ${mem_used}/${mem_total}MB (${mem_percent}%)"
    done
}

# 监控GPU使用情况
monitor_gpus() {
    log_info "开始监控GPU使用情况"
    echo "=== 实时监控GPU使用情况 (按 Ctrl+C 停止) ==="
    echo "日志文件: $LOG_FILE"
    
    trap 'log_info "监控已停止"; echo; return 0' INT
    
    while true; do
        clear
        echo "监控时间: $(date)"
        echo "日志文件: $LOG_FILE"
        echo "================================"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r id name util mem_used mem_total temp; do
            mem_percent=$(awk "BEGIN {printf \"%.1f\", $mem_used/$mem_total*100}")
            echo "GPU $id: $name"
            echo "  温度: ${temp}°C"
            echo "  GPU利用率: ${util}%"
            echo "  内存使用: ${mem_used}/${mem_total}MB (${mem_percent}%)"
            echo ""
        done
        sleep 2
    done
}

# 查看日志
show_logs() {
    local lines=${1:-50}
    echo "=== 最近 $lines 行日志 ==="
    echo "管理器日志: $LOG_FILE"
    echo "Python程序日志: $PYTHON_LOG_FILE"
    echo ""
    echo "--- 管理器日志 ---"
    tail -n "$lines" "$LOG_FILE" 2>/dev/null || echo "日志文件不存在"
    echo ""
    echo "--- Python程序日志 ---"
    tail -n "$lines" "$PYTHON_LOG_FILE" 2>/dev/null || echo "日志文件不存在"
}

# 清除日志
clear_logs() {
    echo "清除日志文件..."
    > "$LOG_FILE"
    > "$PYTHON_LOG_FILE"
    echo "日志已清除"
    log_info "日志已清除"
}

# 主程序
main() {
    # 记录命令执行
    log_info "执行命令: $0 $*"
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        start)
            # 收集所有参数直到遇到选项
            local gpu_spec=""
            local extra_args=""
            local collecting_gpus=true
            
            while [ $# -gt 0 ]; do
                case "$1" in
                    --*)
                        collecting_gpus=false
                        extra_args="$extra_args $1"
                        shift
                        # 如果选项有参数，也收集
                        if [[ "$1" != --* ]] && [ $# -gt 0 ]; then
                            extra_args="$extra_args $1"
                            shift
                        fi
                        ;;
                    *)
                        if [ "$collecting_gpus" = true ]; then
                            gpu_spec="$1"
                        else
                            extra_args="$extra_args $1"
                        fi
                        shift
                        ;;
                esac
            done
            
            start_gpus "$gpu_spec" $extra_args
            ;;
        stop)
            stop_gpus
            ;;
        status)
            show_status
            ;;
        list)
            list_gpus
            ;;
        monitor)
            monitor_gpus
            ;;
        kill)
            kill_all
            ;;
        logs)
            show_logs "$1"
            ;;
        clear-logs)
            clear_logs
            ;;
        help|-h|--help)
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            echo "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主程序
main "$@"