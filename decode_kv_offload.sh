unset ftp_proxy
unset https_proxy
unset http_proxy
export VLLM_VERSION=0.23.0
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
export TASK_QUEUE_ENABLE=1
 
nic_name="enp48s3u1u2" # ifconfig 查看，选和本机 ip 相同的网卡
local_ip=141.61.81.152
 
export ASCEND_BUFFER_POOL=4:8
 
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
 
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
 
rm -rf ~/ascend/log/debug/plog/*
export PYTHONHASHSEED=0
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
 
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export PYTHONPATH=/home/l00889328/offload/vllm:/home/l00889328/offload/vllm-ascend:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"
unset ASCEND_RT_VISIBLE_DEVICES

source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib:$LD_LIBRARY_PATH
export MMC_META_CONFIG_PATH=/home/l00889328/offload/memcache_config/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/home/l00889328/offload/memcache_config/mmc-local.conf
mmc_meta_service &

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
# /mnt/share/GLM-5.2-Provide-0610-W4A8 /home/g00955623/offload/glm-5.2

vllm serve /mnt/share/GLM-5.2-Provide-0610-W4A8 \
    --host 0.0.0.0 \
    --port 8004 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "glm-5" \
    --max-num-seqs 4 \
    --max-model-len 38912 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL", "cudagraph_capture_sizes":[4,8,16,32]}' \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --aggregate-engine-logging \
    --enable-auto-tool-choice \
    --no-enforce-eager \
    --safetensors-load-strategy 'prefetch' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_layerwise": true,
            "layerwise_num_shared_buffers": 4,
            "layerwise_independent_layers": [],
            "connectors": [
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "backend": "memcache",
                        "use_layerwise": true,
                        "save_decode_cache": false,
                        "lookup_rpc_port": "0",
                        "layerwise_num_shared_buffers": 4,
                        "layerwise_independent_layers": []
                    }
                },
                {
                    "kv_connector": "SFAKVOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "use_layerwise": true
                    }
                }
            ]
        }
    }' \
    --additional-config '{
        "enable_cpu_binding":false,
        "use_offload": true,
        "lru_resident_cache_config": {
            "enabled": true,
            "buffer_size": 4096,
            "topk": 2048
        },
        "sfa_kv_offload_cpu_cache_config": {
            "dram_size_gb": 15,
            "cache_budget_ratio": 0.9
        }
    }' \
    2>&1 | tee online_1.log
    # --additional-config '{
    #   "fuse_muls_add": true,
    #   "multistream_overlap_shared_expert": true, "enable_dsa_cp": false,
    #   "enable_npugraph_ex": true, "enable_sparse_c8": false, "enable_cpu_binding":false}' \
    # --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    
