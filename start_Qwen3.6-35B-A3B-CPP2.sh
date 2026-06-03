unset https_proxy
unset http_proxy
nic_name="enp194s0f0" # ifconfig 查看，选和本机 ip 相同的网卡
local_ip=90.90.97.29

export HCCL_IF_IP=$local_ip         # 指定HCCL通信库使用的网卡 IP 地址
export GLOO_SOCKET_IFNAME=$nic_name # 指定使用 Gloo通信库时指定网络接口名称 
export TP_SOCKET_IFNAME=$nic_name   # 指定 TensorParallel使用的网络接口名称
export HCCL_SOCKET_IFNAME=$nic_name # 指定 HCCL 通信库使用的网络接口名称
export OMP_PROC_BIND=false          # 允许操作系统调度线程在多个核心之间迁移
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1          # 在支持 OpenMP 的程序中，最多使用 100 个 CPU 线程进行并行计算
export HCCL_BUFFSIZE=1024           # 每个通信操作的缓冲区大小为 1024 Bytes
# echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# sysctl -w vm.swappiness=0
# sysctl -w kernel.numa_balancing=0
# sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
# export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_VERSION="0.20.2"
export DYNAMIC_EPLB="false"
export ASCEND_RT_VISIBLE_DEVICES=2,3
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=1
# --speculative-config '{"num_speculative_tokens": 1, "method":"mtp"}' \

vllm serve /dev/shm/data/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8007 \
  --served-model-name qwen \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --pipeline_parallel_size 2 \
  --enable-expert-parallel \
  --max-num-seqs 128 \
  --max-model-len 262144  \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  --additional-config '{
    "enable_cpu_binding":true,
    "multistream_overlap_shared_expert": true,
    "profiling_chunk_config":{"enabled":true, "smooth_factor":1.0, "min_chunk":4096}
    }' \
  --profiler_config '{"profiler":"torch", "torch_profiler_dir":"/home/d00822540/profiling/cpp2_k_35B", "torch_profiler_with_stack":false, "torch_profiler_with_memory":false, "torch_profiler_record_shapes":true}' \
  --allowed-local-media-path / \
  --mm-processor-cache-gb 0 \
  --no-async-scheduling \
  --no-enforce-eager \
  --safetensors-load-strategy 'prefetch' \
  --speculative-config '{"num_speculative_tokens": 3, "method":"mtp"}' \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,16,32,64,128], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  2>&1 | tee online.log
