/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef NVRTC_JIT_COMPILATION
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cuda.h>
#endif

namespace deep_gemm
{

#ifndef NVRTC_JIT_COMPILATION

// Structure to hold a set of TMA descriptors for a GEMM operation
struct TMADescriptorSet
{
    CUtensorMap tma_a_desc;
    CUtensorMap tma_b_desc;
    CUtensorMap tma_scales_desc; // can be scales_a or scales_b depending on swapAB
    CUtensorMap tma_d_desc;
};

// Simple cache key based on shape parameters and pointer addresses
struct TMADescriptorKey
{
    void* mat_a;
    void* mat_b;
    void* mat_d;
    void* scales;
    uint32_t shape_m;
    uint32_t shape_n;
    uint32_t shape_k;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t block_k;

    bool operator==(const TMADescriptorKey& other) const
    {
        return mat_a == other.mat_a && mat_b == other.mat_b && mat_d == other.mat_d && scales == other.scales
            && shape_m == other.shape_m && shape_n == other.shape_n && shape_k == other.shape_k
            && block_m == other.block_m && block_n == other.block_n && block_k == other.block_k;
    }
};

struct TMADescriptorKeyHash
{
    std::size_t operator()(const TMADescriptorKey& k) const
    {
        // Simple hash combination
        std::size_t h = 0;
        h ^= std::hash<void*>{}(k.mat_a) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}(k.mat_b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}(k.mat_d) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.shape_m) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.shape_n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.shape_k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Thread-safe singleton cache for TMA descriptors
class TMADescriptorCache
{
public:
    static TMADescriptorCache& getInstance()
    {
        static TMADescriptorCache instance;
        return instance;
    }

    // Check if descriptor set exists in cache
    TMADescriptorSet* get(const TMADescriptorKey& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            return &it->second;
        }
        return nullptr;
    }

    // Insert descriptor set into cache
    TMADescriptorSet* insert(const TMADescriptorKey& key, const TMADescriptorSet& desc_set)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[key] = desc_set;
        return &cache_[key];
    }

    // Clear cache
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

private:
    TMADescriptorCache() = default;
    ~TMADescriptorCache() = default;
    TMADescriptorCache(const TMADescriptorCache&) = delete;
    TMADescriptorCache& operator=(const TMADescriptorCache&) = delete;

    std::unordered_map<TMADescriptorKey, TMADescriptorSet, TMADescriptorKeyHash> cache_;
    std::mutex mutex_;
};

#endif // NVRTC_JIT_COMPILATION

} // namespace deep_gemm
