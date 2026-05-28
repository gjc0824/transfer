# vllm-ascend 最近两个提交代码评审与串讲

评审范围：

- `7bf53043a5f60ca8e641516016c1bbd6ebe20a8f`：`offload the tail buffer to dram`
- `cd60b1e14840434206faa4856efbc002f8ee3815`：`[Presicion] Fix sync stream for KV`
- 合并净 diff：`HEAD~2..HEAD`

总体判断：这两次提交的主线目标是合理的，即把 SFA offload 从“额外 NPU tail window”切到“CPU DRAM 持有完整可寻址 KV，decode 新 token 通过 token-wise patch 更新到 CPU”，并补齐图模式下 KV 保存与 LRU load 的 stream 同步。这个方向可以解释大多数核心改动。但当前仍有一个需要优先修正的元数据传播风险，以及若干可以删回原样的预留接口和 debug 残留。

## 主要发现

### P1：`all_kv_in_cpu` 作为动态属性会在 `unpadded()` 路径丢失

位置：

- `vllm_ascend/worker/model_runner_v1.py:3504-3505`
- `vllm_ascend/attention/utils.py:190-240`
- `vllm_ascend/attention/utils.py:248-301`
- `vllm_ascend/attention/sfa_v1.py:532`

问题说明：

`model_runner_v1.py:3504-3505` 通过 `cm.all_kv_in_cpu = ...` 动态挂载字段，但 `AscendCommonAttentionMetadata` dataclass 本身没有声明这个字段。普通路径下 `getattr(common_attn_metadata, "all_kv_in_cpu", False)` 可以读到它；但 `AscendCommonAttentionMetadata.unpadded()` 会重新构造一个新的 `AscendCommonAttentionMetadata`，当前 `utils.py:248-301` 没有把动态字段传进去。

影响：

当 speculative decode / drafter 触发 `spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)` 时，新的 common metadata 会丢失 `all_kv_in_cpu=True`。随后 `sfa_v1.py:532` 会读到默认 `False`，SFA decode 会走 CPU/NPU 双路合并逻辑，而不是全 CPU 逻辑。轻则多做一次 NPU sparse attention，重则在所有 token 实际都在 CPU 路径时，对空 NPU sparse indices 做 softmax/LSE 合并，存在精度或运行时风险。

建议修复：

- 在 `AscendCommonAttentionMetadata` 中显式增加 `all_kv_in_cpu: bool = False`。
- 在 `unpadded()` 的构造参数中传递 `all_kv_in_cpu=self.all_kv_in_cpu`。
- `model_runner_v1.py:3504-3505` 保留赋值也可以，但最好不再依赖动态属性。

必要性判断：

- `AscendSFAMetadata.all_kv_in_cpu` 是必要的。
- `AscendCommonAttentionMetadata` 侧也应显式增加字段；当前动态挂载不是最小安全改法。

### P2：新增 connector API 中有未使用的预留接口，建议删回去

位置：

- `vllm_ascend/attention/utils.py:461-487`
- `vllm_ascend/attention/utils.py:542-550`
- `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py:314-356`
- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_connector.py:107-125`
- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_connector.py:162-163`
- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py:594-600`
- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py:645-697`

问题说明：

当前实际 decode CPU patch 走的是 `prepare_lru_resident_and_load(..., key_cache, value_cache, slot_mapping, positions, num_decode_tokens)` 这条路径；`maybe_update_cpu_kv_tokens()`、connector 的 `update_cpu_kv_tokens()`、`maybe_get_num_cpu_blocks()`、`get_num_cpu_blocks()` 在仓库内没有调用点。

影响：

这些接口扩大了 connector 协议面，增加维护成本，而且会让读代码的人误以为存在另一条 CPU KV 更新路径。`cpu_blocks_by_req` 目前也只服务于未使用的 `get_num_cpu_blocks()`。

建议修复：

- 如果没有外部调用方，删除 `maybe_update_cpu_kv_tokens()`、`update_cpu_kv_tokens()` 透传链路。
- 如果没有外部调用方，删除 `maybe_get_num_cpu_blocks()`、`get_num_cpu_blocks()` 和 `cpu_blocks_by_req`。
- 保留 `ensure_layer_saved()`，它在非图全 CPU decode 路径中是必要同步点。

必要性判断：

- `ensure_layer_saved()` 必要。
- `update_cpu_kv_tokens()` / `get_num_cpu_blocks()` 当前不必要。

### P3：`ACLGraphEntry.replay_count` 已无实际用途

位置：

- `vllm_ascend/compilation/acl_graph.py:59`
- `vllm_ascend/compilation/acl_graph.py:260`

问题说明：

第一个提交加入了图 replay 调试日志，第二个提交删除了调试日志，但留下了 `replay_count` 字段和自增。当前没有任何读取点。

建议修复：

- 删除 `replay_count` 字段和 `entry.replay_count += 1`。

必要性判断：

- 若不保留 debug 采样日志，此改动不必要。

### P3：全 CPU decode 依赖 `positions`，但当前缺失时会静默跳过 token patch

位置：

- `vllm_ascend/attention/sfa_v1.py:1604-1620`
- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py:953-1014`

问题说明：

`_get_topk_buffer()` 在全 CPU 路径下也允许 `attn_metadata.positions is None`，此时传给 worker 的 `positions` 为 `None`，worker 不会构造 `token_update_args`，decode 新 token 就不会 patch 到 CPU cache。通常 SFA metadata 会有 positions，但这里是 correctness 关键依赖，建议显式断言。

建议修复：

在 `all_kv_in_cpu` 分支或调用 connector 前加入：

```python
if all_kv_in_cpu and attn_metadata.positions is None:
    raise RuntimeError("SFA all-CPU offload requires positions metadata")
```

必要性判断：

- 不是必须重构，但建议加防御性检查。

## 提交级串讲

### 提交 7bf53043：offload the tail buffer to dram

核心变化：

- 移除 tail window NPU cache 方案。
- 让 scheduler 为尾块也分配 CPU block。
- worker 在 CPU LRU load 前，把 decode 新 token 从 real KV cache patch 到 CPU pool。
- attention decode 从“CPU 历史 + NPU tail window 合并”改成“全 CPU 或 CPU/NPU 按 offload threshold 分流”。
- full graph + sequence parallel 场景补充可达的 decode graph key。

必要性判断：

- 为了解决 tail buffer 持久化到 DRAM，scheduler/worker/attention 三处协议必须一起改。
- 第一次提交里夹带的大量 `SFA_DEBUG` / `SFA_PROBE` / graph debug 日志不是最终功能所需，第二次提交已删除，方向正确。

### 提交 cd60b1e：Fix sync stream for KV

核心变化：

- 删除第一版中的调试环境变量、probe、日志和强制 LRU miss 逻辑。
- 保留并强化 `save_cpu()` 前的 stream event 等待。
- 图模式 host func 中在 pre-attention load 前触发并等待本层 KV save。
- 删除 `kv_transfer.py` 中临时 debug summary，最终该文件净 diff 为 0。

必要性判断：

- 删除调试代码必要。
- `save_cpu()` 中的 `current_stream.record_event()` 和 `save_stream.wait_event()` 必要，因为后台 save stream 读 HBM 前必须等当前计算流完成 real KV 写入。
- 留下的 `replay_count` 不必要。

## 逐文件逐段串讲

### `vllm_ascend/attention/sfa_v1.py`

#### `130-134`：`build_valid_topk_mask`

- `130-133` 定义 topk 合法性 helper。
- `134` 同时过滤 `topk_indices >= 0` 和 `topk_indices < seq_len_thresholds`。
- 必要性：逻辑必要；是否抽成函数可商榷。当前只用一次，内联也可以；保留有助于表达“合法 topk”的语义。

#### `197-245`：`AscendSFAMetadata` 新增 `all_kv_in_cpu`

- `241-244` 已有 offload 相关 per-request/per-token metadata。
- `245` 新增 `all_kv_in_cpu`，用于 decode 时选择全 CPU attention 还是 CPU/NPU 双路合并。
- 必要性：必要。没有这个 flag，attention 侧无法区分“所有 topk 都从 CPU LRU buffer 读”和“旧 token 从 CPU、尾部仍从 NPU 读”。
- 风险：上游 common metadata 没有显式字段，见 P1。

#### `1492-1646`：`_get_topk_buffer`

- `1501` 取 forward context，用于判断 graph runtime / capturing。
- `1502` `num_tokens` 是 decode topk 行数，MTP/spec decode 时可能大于请求数。
- `1503` `num_reqs` 是 decode 请求数。
- `1504-1507` 对无 decode 请求、缺 `req_ids_tensor` 做硬错误，必要。
- `1509-1510` 从 `kv_cache[3:5]` 取 CPU/LRU resident topk buffer，必要。
- `1511` squeeze topk 的中间维，后面按 `[rows, topk]` 处理，必要。
- `1513-1525` 区分 MTP 与普通 decode。MTP 必须使用 `token_to_req` 将每个 decode token 映射回请求行，必要。
- `1527-1530` 用 seq len 构造合法 topk mask，避免加载越界 token，必要。
- `1531` 判断当前是否 graph runtime。这里把 replay 也视为 graph path，目的是让 worker 走 host func；必要。
- `1533-1542` 全 CPU 分支：非图模式先 `ensure_layer_saved()`，然后所有合法 topk 都交给 CPU LRU。必要。
- `1543-1597` 非全 CPU 分支：按 `num_offloaded_blocks * block_size` 拆成 NPU 和 CPU 两部分。这个分支保留了兼容性，但在当前 model runner 总是设置 `all_kv_in_cpu=True` 后，主要是 fallback。
- `1551-1590` side stream 上执行 NPU sparse attention。必要性取决于非全 CPU fallback 是否仍需要；若确定 all-CPU 是唯一模式，可后续收敛。
- `1599-1603` MTP 下 req id 也要按 token 行展开，必要。
- `1604-1620` 调 worker 准备 LRU resident buffer，并把 real KV cache、slot mapping、positions 传下去做 token-wise CPU patch。必要。
- `1622-1627` 把 topk buffer 视为 PA_BSND block layout，必要。
- `1628-1633` compact resident slot id，把有效 slot 排到前面，符合 SFA op 输入要求，必要。
- `1635-1636` 非全 CPU 才等待 side stream，必要。
- `1637-1645` 返回 CPU buffer、indices、固定 sparse block table，以及可选 NPU attention 输出/LSE。`_cpu_mask` 当前调用方不用，可简化。

#### `1708-1810`：decode attention 执行

- `1711-1716` 把 decode 和 prefill token 拆开，沿用原结构，必要。
- `1724-1742` decode 存在时调用 `_get_topk_buffer()`，必要。
- `1743-1762` 全 CPU 分支只对 resident buffer 跑一次 sparse attention，不做 LSE merge，必要。
- `1763-1810` 非全 CPU 分支分别算 CPU 与 NPU attention，再用 `npu_attention_update` 合并，保留兼容性。
- 必要性：全 CPU 分支是本次 DRAM tail offload 的核心；非全 CPU分支若没有计划继续支持，可以后续评估是否删除。

#### 删除 tail window 相关代码

删除内容：

- `tail_window_blocks`
- `tail_block_table`
- `tail_block_offsets`
- `_maybe_update_sfa_tail_cache()`
- `len(kv_cache) >= 7` 判定改成 `>= 5`

必要性：必要。最终 KV tuple 不再包含 `[5:7]` tail cache，继续保留会浪费 NPU 显存并维持两套 tail 数据源。

### `vllm_ascend/attention/utils.py`

#### `451-458`：`maybe_ensure_kv_layer_saved_to_connector`

- `452-453` 没有 v1 connector 时直接返回。
- `455-458` 有 hook 时调用。
- 必要性：必要。非图全 CPU decode 在 LRU load 前必须确保当前层待保存的 prefill/chunk KV 已经进 CPU pool。

#### `461-487`：`maybe_update_cpu_kv_tokens`

- 当前仓库内无调用点。
- 必要性：当前不必要。实际 token patch 已合并进 `maybe_prepare_lru_resident_and_load_graph()` 的参数中。

#### `500-539`：扩展 `maybe_prepare_lru_resident_and_load_graph`

- `509-513` 新增 real KV cache、slot mapping、positions、decode token 数。
- `525-539` 原样透传给 connector。
- 必要性：必要。图模式和非图模式都通过这个统一入口完成 LRU compact、CPU token patch、H2D batch copy。

#### `542-550`：`maybe_get_num_cpu_blocks`

- 当前仓库内无调用点。
- 必要性：当前不必要，建议删除。

### `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/config_data.py`

#### `99-107`：`RequestTracker` 新增显式 offload 映射

- `104-107` 保存本调度步实际要从哪些 HBM block 复制到哪些 CPU block。
- 必要性：必要。尾块可能重复刷新，不能再简单用“最后 N 个 block”推导。

#### `119-143`：`ReqMeta` 透传 offload 映射

- `125-128` metadata 中携带显式 HBM->CPU 映射。
- `141-142` `from_request_tracker()` 负责传下去。
- 必要性：必要。scheduler 和 worker 必须共享同一个映射，否则尾块刷新会错位。

### `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_scheduler.py`

#### `31-32`：`_num_covered_blocks`

- 用 ceil 计算 token 覆盖的 block 数。
- 必要性：必要。tail block 不是满 block，但仍需要 CPU block 承载。

#### `98-100`：`decode_width`

- 普通 decode 为 1；spec decode 加 draft token 数。
- 必要性：必要。scheduler 需要判断当前 step 是 decode 还是 chunk/prefill。

#### `170-190`：新请求分配 CPU block

- `171` 取 real KV group 的 NPU block table。
- `172-174` 用已计算 token + 本步 finalized token 计算覆盖 block。
- `175` 分配 CPU blocks。
- `180-181` 明确本步要保存的 HBM/CPU block。
- 必要性：必要。新请求首个 prefill/chunk 后，tail block 也应保存到 CPU。

#### `213-253`：cached request 更新

- `213-215` 计算本步后 token/block 覆盖范围。
- `217-222` 计算目标 CPU block 数，只为新增覆盖 block 分配 CPU block。
- `224-227` 判定 decode step。
- `228-234` decode step 不做整块刷新，避免 indexer alias 污染 historical real KV；改由 attention 后的 token-wise patch 写 CPU。
- `235-239` chunk/prefill step 从被本 chunk 触及的 block 开始刷新，必要。
- `240-241` 无新 token 时只保存新分配部分。
- `242-253` 校验并写入显式 HBM->CPU 映射。
- 必要性：这段是 tail-to-DRAM 协议的核心，整体必要。

### `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py`

#### `463-496`：token update staging buffer

- 分别为 slot、position、token_to_req、K、V 准备 pinned CPU buffer。
- 必要性：必要。图模式 host func 需要 CPU 侧可访问的稳定 buffer；同时注释说明避免 D2H+dtype conversion 同步 graph stream。

#### `602-619`：`save_cpu`

- `615-616` 记录当前 compute stream event，并让 save stream 等待。
- `617-619` 提交本层 save task。
- 必要性：必要。这是第二个提交“Fix sync stream for KV”的关键修正，防止 save stream 在 real KV 写完前读 HBM。

#### `626-643`：`ensure_layer_saved`

- 非图模式下，如果本层存在 save task，则提交并等待完成。
- 必要性：必要。全 CPU decode 在读 CPU pool 前必须看到本层最新 KV。
- 可简化点：`633-635` 的 event wait 与 `save_cpu()` 内部 `615-616` 有重复，保留不影响 correctness，但可以删掉外层重复等待，减少认知负担。

#### `645-697`：`update_cpu_kv_tokens`

- 当前没有调用点。
- 必要性：当前不必要。真正使用的是 `_update_cpu_kv_tokens_from_cpu()` 和 `prepare_lru_resident_and_load()` 中的 `token_update_args`。

#### `699-765`：`_update_cpu_kv_tokens_from_cpu`

- `719` 使用 CPU block table。
- `720-757` 对每个 decode token 计算目标 CPU block 和 block 内 offset。
- `762-764` 把 K/V token 写到 CPU cache。
- 必要性：必要。这是 decode tail 进入 DRAM 的 token-wise patch 核心。
- 风险：严格模式会在缺 CPU block 时抛错；host func 路径用 `strict=False`，直接路径用 `strict=True`。如果删除未使用直接路径，可以减少分支。

#### `783-904`：host func 主体

- `819-847` 先做 token-wise CPU patch。
- `848-869` 执行 LRU resident compact。
- `870-888` 计算 CPU->HBM batch copy 地址。
- `893-904` 图模式下在 host func 中触发并等待本层 save，避免 graph replay 外二次打断。
- 必要性：必要。顺序上先 patch 再 compact/load，保证 decode 新 token 可被本步 topk 命中。

#### `906-1079`：`prepare_lru_resident_and_load`

- `922` 合并显式 `capturing` 和实际 stream capture 状态。
- `938-948` 准备 CPU block table；MTP 时按 token_to_req 展开。
- `949-952` D2H topk 和 req id。
- `953-1014` 如果传入 real KV 信息，则准备 token patch 参数。
- `1052-1064` 图模式通过 `_launch_host_func` 保证 CPU 逻辑挂在当前 stream 顺序中。
- `1068-1075` 把 batch copy 参数传到 NPU 并执行 H2D copy。
- `1077-1078` 把 LRU 当前 slot 回写给 attention。
- 必要性：必要。

#### `1081-1094`：`process_layer_data`

- `1086-1088` 没有新 offload block 直接返回。
- `1089-1090` 优先使用 scheduler 显式映射。
- `1091-1094` 映射长度不一致直接抛错。
- 必要性：必要。原来的“如果 NPU 多一个未满 block 就裁掉”不再适用于 tail block offload。

### `vllm_ascend/worker/model_runner_v1.py`

#### `1445-1478`：构造 offload runtime metadata

- `1446-1452` 从 scheduled token 中扣掉 draft token，得到 finalized scheduled token。必要。
- `1454-1458` 设置 `all_kv_in_cpu=True`，表达当前 buffer mixed SFA offload 设计目标：decode 可从 CPU 读到全部 KV。必要。
- `1459-1464` 用 ceil 计算 covered CPU block 数，与 scheduler tail block 分配一致。必要。
- `1465-1469` prefill/chunk batch 将 `num_offloaded_blocks` 置 0。对全 CPU分支不是核心，但作为 fallback metadata 可保留。
- `1471-1478` 将 offload metadata 拷到 NPU buffer，必要。

#### `3504-3505`：把 `all_kv_in_cpu` 写入 common metadata

- 必要性：必要。
- 问题：当前是动态属性，应改为 dataclass 显式字段并让 `unpadded()` 传播，见 P1。

#### `4779-4791`：删除 tail NPU cache tuple

- 删除 `tail_k_cache` / `tail_v_cache` 分配。
- KV tuple 从 7 段收敛回 5 段：`k_cache, v_cache, dsa_k_cache, topk_buffer_k, topk_buffer_v`。
- 必要性：必要。tail 已进入 DRAM，不应再额外占用 NPU cache。

#### `5502-5548`：override `_warmup_and_capture`

- `5513-5523` full decode + uniform + SP 时，用真实 decode token 数跑 dummy，再让 dispatcher pad 回 graph size。
- `5525-5548` warmup 和 capture 均走 `_dummy_run()`。
- 必要性：看起来必要，用于配合 `patch_cudagraph.py` 中新增的 SP padded decode graph key。
- 风险：这是复制/覆盖上游 runner 方法，后续 vLLM 升级时容易漂移；建议加一条针对 SP full decode graph capture 的测试。

### `vllm_ascend/patch/worker/patch_cudagraph.py`

#### `20-32`：判断是否启用 sequence parallel padding

- 依次检查 pass config、additional_config、环境变量。
- 必要性：必要。只有 SP/flashcomm bypass 场景才需要额外 decode capture sizes。

#### `67-92`：计算可达 decode capture sizes

- `83-85` 按请求数枚举 raw decode token，再按 TP size round up。
- `88-90` 查 padded graph size 并过滤到 capture sizes。
- 必要性：必要。
- 建议：如果 `_bs_to_padded_graph_size` 不保证覆盖所有 `dispatch_tokens <= max_size`，`88` 可能 KeyError。更稳妥是用 `.get()` 或先判断 key 存在。

#### `95-129`：patch `initialize_cudagraph_keys`

- 先调用原始初始化，再给 FULL separate decode 额外注册 SP 可达 key。
- 必要性：必要。

### `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`

- `268-306` 扩展 `prepare_lru_resident_and_load()` 参数并转发，必要。
- `308-312` 转发 `ensure_layer_saved()`，必要。
- `314-340` 转发 `update_cpu_kv_tokens()`，当前不必要。
- `348-356` 转发 `get_num_cpu_blocks()`，当前不必要。

### `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_connector.py`

- `104-105` 转发 `ensure_layer_saved()`，必要。
- `127-157` 扩展 `prepare_lru_resident_and_load()` 参数，必要。
- `107-125` `update_cpu_kv_tokens()` 当前不必要。
- `162-163` `get_num_cpu_blocks()` 当前不必要。
- `165-168` finished 时清理 worker 里的 req 状态；如果删除 `cpu_blocks_by_req`，这行也可以简化。

### `vllm_ascend/compilation/acl_graph.py`

- `59` `replay_count` 和 `260` 自增是 debug 残留。
- 必要性：不必要，建议删除。

### `vllm_ascend/core/single_type_kv_cache_manager.py`

- 净变化只是注释。
- `302-305` 注释从 tail buffer 改成 CPU pool/normal HBM allocation 语义，和最终设计一致。
- `359` 附近删除的注释不影响行为。
- 必要性：可保留，但不是功能必要；如果想最小 diff，可以只保留能澄清当前设计的第一段注释。

### `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/kv_transfer.py`

- 第一个提交新增 `_SFA_DEBUG` 和 save 后 summary。
- 第二个提交完全删除。
- 净 diff 为 0。
- 必要性：最终状态正确，不应保留临时 debug。

## 建议的最小修正清单

1. 修复 `all_kv_in_cpu` 传播：

```python
@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    ...
    all_kv_in_cpu: bool = False

def unpadded(...):
    return AscendCommonAttentionMetadata(
        ...
        all_kv_in_cpu=self.all_kv_in_cpu,
    )
```

2. 删除未使用接口：

- `maybe_update_cpu_kv_tokens`
- `maybe_get_num_cpu_blocks`
- `AscendMultiConnector.update_cpu_kv_tokens`
- `AscendMultiConnector.get_num_cpu_blocks`
- `SFAKVOffloadConnector.update_cpu_kv_tokens`
- `SFAKVOffloadConnector.get_num_cpu_blocks`
- `SFAKVOffloadWorker.update_cpu_kv_tokens`
- `SFAKVOffloadWorker.get_num_cpu_blocks`
- `SFAKVOffloadWorker.cpu_blocks_by_req`

3. 删除 ACL graph replay debug 残留：

- `ACLGraphEntry.replay_count`
- `entry.replay_count += 1`

4. 给全 CPU path 加 positions 断言。

5. 给 `_reachable_decode_capture_sizes()` 的 dict lookup 加保护，或确认 `_bs_to_padded_graph_size` 覆盖完整。

## 验证记录

已执行：

```bash
git diff --check HEAD~2..HEAD
python -m py_compile <11 个修改文件>
ruff check <11 个修改文件>
```

结果：

- `git diff --check` 通过。
- `py_compile` 通过。
- `ruff check` 未通过，报告 38 个问题；其中大量是该文件既有风格问题或第一版调试遗留附近的格式问题。若要合入主干，建议单独跑一次 `ruff check --fix` 或只对本次新增行做最小修复，避免把大范围格式化混进功能提交。

## 建议补充测试

1. SFA offload + 普通 decode：topk 命中历史 token 和当前 decode token，验证 CPU patch 后输出与不开 offload 对齐。
2. SFA offload + mixed prefill/decode：prefill/chunk save 与 decode LRU load 同层发生，验证 `ensure_layer_saved()` 和 host func save 顺序。
3. SFA offload + speculative decode：触发 `spec_decode_common_attn_metadata.unpadded()`，验证 `all_kv_in_cpu` 不丢。
4. Full graph + TP>1 + sequence parallel：decode token 数不是 TP size 整数倍，验证新增 graph key 能 capture/replay。
5. final chunk prefill token 数小于等于 decode width：验证 scheduler 不误判为 decode 整块刷新。
