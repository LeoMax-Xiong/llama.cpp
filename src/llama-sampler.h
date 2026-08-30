#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// 采样器链：按添加顺序保存一组采样器，采样时依次应用。
// 采样器链的运行设备（is_backend 为 true）运行在后端（GPU）采样图中，
// 其余采样器在 CPU 上依次执行。
struct llama_sampler_chain {
    llama_sampler_chain_params params; // 采样器链参数（如是否统计性能耗时）

    // 是否已调用 .backend_init() 完成后端采样图初始化
    bool is_init = false;

    uint32_t n_nodes = 0; // 后端采样图的节点数

    struct info {
        bool is_backend;      // 该采样器是否为后端采样器（运行在 GPU 上）
        llama_sampler * ptr;  // 指向链内的采样器
    };

    std::vector<info> samplers; // 链内采样器列表，按添加顺序排列

    // llama_sampler_sample 预分配的候选缓冲区，避免每次采样重复分配
    std::vector<llama_token_data> cur;

    // 采样性能统计
    mutable int64_t t_sample_us; // 累计采样耗时（微秒）
    mutable int32_t n_sample;    // 累计采样次数
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
