#!/bin/bash

# 自动拉取代码脚本
# 每隔5分钟自动执行 git pull origin main

# 设置脚本名称和日志文件
SCRIPT_NAME="auto_pull.sh"
LOG_FILE="auto_pull.log"

# 日志函数
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$LOG_FILE"
}

# 检查是否在git仓库中
check_git_repo() {
    if [ ! -d ".git" ]; then
        log_message "错误: 当前目录不是git仓库"
        exit 1
    fi
    log_message "确认在git仓库中"
}

# 检查当前分支
check_current_branch() {
    local current_branch=$(git branch --show-current 2>/dev/null)
    if [ -n "$current_branch" ]; then
        log_message "当前分支: $current_branch"
        if [ "$current_branch" != "main" ]; then
            log_message "警告: 当前不在main分支，而是 $current_branch 分支"
        fi
    else
        log_message "无法获取当前分支信息"
    fi
}

# 执行git pull
execute_git_pull() {
    log_message "开始执行 git pull origin main..."
    
    # 执行git pull
    if git pull origin main 2>&1 | tee -a "$LOG_FILE"; then
        log_message "Git pull 成功完成"
    else
        log_message "Git pull 执行失败"
    fi
}

# 主函数
main() {
    log_message "启动自动拉取代码脚本"
    log_message "脚本将每5分钟执行一次 git pull origin main"
    
    # 检查环境
    check_git_repo
    check_current_branch
    
    # 主循环
    while true; do
        log_message "=" * 50
        log_message "执行时间: $(date)"
        
        execute_git_pull
        
        log_message "等待5分钟后再次执行..."
        log_message "=" * 50
        
        # 等待5分钟
        sleep 300
    done
}



# 捕获中断信号
trap 'log_message "收到中断信号，正在退出..."; exit 0' INT TERM

# 运行主函数
main
