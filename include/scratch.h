// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "boost_dynamic_bitset_fwd.h"
//#include "boost/dynamic_bitset.hpp"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "tsl/sparse_map.h"

#include "neighbor.h"
#include "concurrent_queue.h"
#include "pq.h"
#include "aligned_file_reader.h"

#define MAX_GRAPH_DEGREE 512
#define MAX_N_CMPS 16384
#define SECTOR_LEN (_u64) 4096
#define MAX_N_SECTOR_READS 128

namespace diskann {
  //
  // Scratch space for in-memory index based search
  //
  template<typename T>
  class InMemQueryScratch {
   public:


    ~InMemQueryScratch();
    InMemQueryScratch(uint32_t search_l, uint32_t indexing_l, uint32_t r,
                      uint32_t maxc, size_t dim, bool init_pq_scratch = false);
    void resize_for_new_L(uint32_t new_search_l);
    void clear();

    inline uint32_t get_L() {
      return _L;
    }
    inline uint32_t get_R() {
      return _R;
    }
    inline uint32_t get_maxc() {
      return _maxc;
    }

    inline T *aligned_query() {
      return _aligned_query;
    }
    inline PQScratch<T> *pq_scratch() {
      return _pq_scratch;
    }

    inline std::vector<Neighbor> &pool() {
      return _pool;
    }
    inline tsl::robin_set<unsigned> &visited() {
      return _visited;
    }
    inline NeighborPriorityQueue &best_l_nodes() {
      return _best_l_nodes;
    }
    inline std::vector<float> &occlude_factor() {
      return _occlude_factor;
    }

    inline tsl::robin_set<unsigned> &inserted_into_pool_rs() {
      return _inserted_into_pool_rs;
    }
    inline boost::dynamic_bitset<> &inserted_into_pool_bs() {
      return *_inserted_into_pool_bs;
    }
    inline std::vector<unsigned> &id_scratch() {
      return _id_scratch;
    }
    inline float *dist_scratch() {
      return _dist_scratch.data();
    }

    inline uint32_t *indices() {
      return _indices.data();
    }
    inline float *interim_dists() {
      return _interim_dists.data();
    }

   private:
    uint32_t _L;
    uint32_t _R;
    uint32_t _maxc;
    
    T *_aligned_query = nullptr;

    PQScratch<T> *_pq_scratch = nullptr;

    std::vector<Neighbor>    _pool;
    tsl::robin_set<unsigned> _visited;
    NeighborPriorityQueue    _best_l_nodes;
    std::vector<float>       _occlude_factor;

    tsl::robin_set<unsigned> _inserted_into_pool_rs;
    boost::dynamic_bitset<> *_inserted_into_pool_bs;

    std::vector<unsigned> _id_scratch;
    std::vector<float>    _dist_scratch;

    std::vector<uint32_t> _indices;        // only used by search
    std::vector<float>    _interim_dists;  // only used by search
  };

  //
  // Scratch space for SSD index based search
  //

  template<typename T>
  class SSDQueryScratch {
   public:
    T   *coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    T *aligned_query_T = nullptr;

    PQScratch<T> *_pq_scratch;

    tsl::robin_set<_u64>  visited;
    NeighborPriorityQueue retset;
    std::vector<Neighbor> full_retset;

    SSDQueryScratch(size_t aligned_dim, size_t visited_reserve);
    ~SSDQueryScratch();

    void reset();
  };

  template<typename T>
  class SSDThreadData {
   public:
    SSDQueryScratch<T> scratch;
    IOContext          ctx;

    SSDThreadData(size_t aligned_dim, size_t visited_reserve);
    void clear();
  };

  //
  // Class to avoid the hassle of pushing and popping the query scratch.
  //
  template<typename T>
  class ScratchStoreManager {
   public:
    ScratchStoreManager(ConcurrentQueue<T *> &query_scratch)
        : _scratch_pool(query_scratch) {
      _scratch = query_scratch.pop();
      while (_scratch == nullptr) {
        query_scratch.wait_for_push_notify();
        _scratch = query_scratch.pop();
      }
    }
    T *scratch_space() {
      return _scratch;
    }

    ~ScratchStoreManager() {
      _scratch->clear();
      _scratch_pool.push(_scratch);
      _scratch_pool.push_notify_all();
    }

    void destroy() {
      while (!_scratch_pool.empty()) {
        auto scratch = _scratch_pool.pop();
        while (scratch == nullptr) {
          _scratch_pool.wait_for_push_notify();
          scratch = _scratch_pool.pop();
        }
        delete scratch;
      }
    }

   private:
    T                    *_scratch;
    ConcurrentQueue<T *> &_scratch_pool;
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
  };
}  // namespace diskann
