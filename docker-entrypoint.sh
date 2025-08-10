#!/bin/sh
set -e

# 设置默认的用户ID和组ID，如果环境变量没有提供
USER_ID=${LOCAL_USER_ID:-1000}
GROUP_ID=${LOCAL_GROUP_ID:-1000}

# 创建与主机用户具有相同ID的用户和组
# 使用-o允许非唯一ID
groupadd -g "$GROUP_ID" -o user
useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user
usermod -aG root user

# 设置HOME环境变量，以确保工具（如pip）在正确的用户目录中工作
export HOME=/home/user

# 使用gosu切换到新创建的用户，并执行传递给脚本的任何命令（即Dockerfile中的CMD）
exec /usr/local/bin/gosu user "$@" 