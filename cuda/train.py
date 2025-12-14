#!/usr/bin/env python3
# enhanced_gpu_stress.py
import torch
import sys
import os
from threading import Thread
import argparse
import time

def stress_gpu(device_id, duration=None, memory_ratio=0.95):
    """
    压力测试指定GPU
    
    Args:
        device_id: GPU设备ID
        duration: 运行时长（秒），None表示无限运行
        memory_ratio: 内存占用比例
    """
    device = torch.device(f'cuda:{device_id}')
    total_mem = torch.cuda.get_device_properties(device).total_memory
    reserve_bytes = int((1 - memory_ratio) * total_mem)
    
    block_size = 1024 * 1024 * 10  # 10MB
    
    blocks = []
    allocated_bytes = 0
    
    print(f"[GPU {device_id}] 开始分配内存...")
    print(f"[GPU {device_id}] 总内存: {total_mem/1024/1024/1024:.2f} GB")
    print(f"[GPU {device_id}] 目标占用: {memory_ratio*100:.1f}%")
    
    # 内存分配阶段
    start_time = time.time()
    while True:
        # 检查运行时长
        if duration and (time.time() - start_time) > duration:
            break
            
        try:
            if allocated_bytes + block_size*4 > total_mem - reserve_bytes:
                break
            x = torch.empty(int(block_size), dtype=torch.float32, device=device)
            blocks.append(x)
            allocated_bytes += x.element_size() * x.nelement()
        except Exception as e:
            print(f"[GPU {device_id}] 内存分配停止: {e}")
            break
    
    if allocated_bytes > 0:
        print(f"[GPU {device_id}] 已分配 {(allocated_bytes/1024/1024):.1f} MB "
              f"({allocated_bytes/total_mem*100:.1f}%)")
        
        # 计算阶段
        print(f"[GPU {device_id}] 开始计算压力测试...")
        a = blocks[0]
        
        compute_start = time.time()
        iterations = 0
        
        try:
            while True:
                # 检查运行时长
                if duration and (time.time() - start_time) > duration:
                    break
                    
                for b in blocks[1:]:
                    a.add_(b)
                iterations += 1
                
                # 每1000次迭代报告一次
                if iterations % 1000 == 0:
                    elapsed = time.time() - compute_start
                    print(f"[GPU {device_id}] 运行 {elapsed:.1f} 秒, "
                          f"迭代次数: {iterations}")
                        
        except KeyboardInterrupt:
            print(f"[GPU {device_id}] 计算被中断")
    else:
        print(f"[GPU {device_id}] 无法分配内存")

def get_available_gpus():
    """获取可用的GPU列表"""
    try:
        return list(range(torch.cuda.device_count()))
    except:
        return []

def get_unused_gpus():
    """获取未被使用的GPU列表（简化实现）"""
    # 这里简化处理，实际应用中可能需要检查具体的GPU使用情况
    return get_available_gpus()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU压力测试工具')
    parser.add_argument('gpus', nargs='?', default='all', 
                       help='要使用的GPU: "all", "remaining", "0,1,2", 或 "0-2"')
    parser.add_argument('--duration', type=int, default=None,
                       help='运行时长（秒）')
    parser.add_argument('--memory-ratio', type=float, default=0.95,
                       help='内存占用比例 (0.0-1.0)')
    parser.add_argument('--list-gpus', action='store_true',
                       help='列出可用GPU')
    
    args = parser.parse_args()
    
    # 列出GPU信息
    if args.list_gpus:
        available_gpus = get_available_gpus()
        print(f"可用GPU: {available_gpus}")
        print(f"GPU总数: {len(available_gpus)}")
        if available_gpus:
            for i in available_gpus:
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory/1024/1024/1024:.1f} GB)")
        sys.exit(0)
    
    # 解析GPU参数
    gpu_list = []
    
    if args.gpus == 'all':
        gpu_list = get_available_gpus()
        print(f"使用所有可用GPU: {gpu_list}")
    elif args.gpus == 'remaining':
        gpu_list = get_unused_gpus()
        print(f"使用剩余GPU: {gpu_list}")
    elif '-' in args.gpus:
        # 范围格式: 0-3
        start, end = map(int, args.gpus.split('-'))
        gpu_list = list(range(start, end + 1))
    elif ',' in args.gpus:
        # 逗号分隔: 0,1,2
        gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        # 单个GPU
        gpu_list = [int(args.gpus)]
    
    # 验证GPU列表
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("错误: 未检测到可用GPU")
        sys.exit(1)
    
    invalid_gpus = [gpu for gpu in gpu_list if gpu not in available_gpus]
    if invalid_gpus:
        print(f"错误: 无效的GPU ID: {invalid_gpus}")
        print(f"可用GPU: {available_gpus}")
        sys.exit(1)
    
    if not gpu_list:
        print("错误: 未指定有效的GPU")
        sys.exit(1)
    
    print(f"开始在 {len(gpu_list)} 个GPU上进行压力测试: {gpu_list}")
    print(f"内存占用比例: {args.memory_ratio*100:.1f}%")
    if args.duration:
        print(f"运行时长: {args.duration} 秒")
    
    # 创建并启动线程
    threads = []
    for gpu_id in gpu_list:
        t = Thread(target=stress_gpu, args=(gpu_id, args.duration, args.memory_ratio))
        t.start()
        threads.append(t)
    
    # 等待线程完成
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print('\n接收到中断信号，正在退出...')
        sys.exit(0)