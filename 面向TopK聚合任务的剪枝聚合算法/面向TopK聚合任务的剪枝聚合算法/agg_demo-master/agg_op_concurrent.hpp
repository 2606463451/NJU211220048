#pragma once
#include <math.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "profiler.h"
#include "row.hpp"

/**
 * @brief 多线程执行
 *
 * @param thread_func 线程函数，格式必须为:
 * auto thread_func = [&](int64_t thread_id, int64_t row_begin, int64_t row_end)
 *
 * @param n_threads 期望线程数
 * @param n_rows 要处理的数据行数
 */
#define CONCURRENT_EXEC_FUNC(thread_func, n_threads, n_rows)            \
  if (n_threads > 1) {                                                  \
    int64_t delta = std::ceil(static_cast<double>(n_rows) / n_threads); \
    std::vector<std::thread> threads;                                   \
    for (int64_t i = 0; i < n_threads; ++i) {                           \
      int64_t row_begin = i * delta;                                    \
      int64_t row_end = (i + 1) * delta;                                \
      row_end = row_end < n_rows ? row_end : n_rows;                    \
      threads.emplace_back(thread_func, i, row_begin, row_end);         \
    }                                                                   \
    for (int i = 0; i < n_threads; ++i) {                               \
      threads[i].join();                                                \
    }                                                                   \
  } else {                                                              \
    thread_func(0, 0, n_rows);                                          \
  }

/**
 * ConcurrentTopKAgg算子的基类
 * - 只实现了group by(int64_t) + count(*) + order by(count(*)) + limit的功能
 */
class ConcurrentTopKAggregateOp {
 public:
  ConcurrentTopKAggregateOp(int64_t n_threads) : n_threads_(n_threads) {}

  // 将单列(int64_t)的数据输入，输出双列(group_key,count(*))的topk聚合结果
  void run(Chunk &input_chunk, Chunk &res_chunk);
  virtual void run_impl(Chunk &input_chunk, Chunk &res_chunk) = 0;

  // 返回算子运行时间
  double get_runtime() { return profiler_.TotalTime(); }

 protected:
  // 定义了用于聚合的哈希表
  typedef std::unordered_map<int64_t, int64_t> KEYCOUNT_HT;

  // 定义了用于求TopK的小顶堆
  struct Item {
    int64_t key_;
    int64_t count_;
    bool operator()(const Item &l, const Item &r) {
      if (l.count_ == r.count_) {
        return l.key_ > r.key_;
      }
      return l.count_ > r.count_;
    }
  };
  typedef std::priority_queue<Item, std::vector<Item>, Item> TOPK_HEAP;

  /**
   * @brief 负责将哈希表转换为topk_heap
   * @param[out] topkheap
   */
  void convert_ht2topkheap(KEYCOUNT_HT &ht, TOPK_HEAP &topk_heap);

  // 负责将小顶堆转换为算子输出结果
  void convert_heap2chunk(TOPK_HEAP &heap, Chunk &chunk);

  inline int64_t hash(int64_t key) { return key * 2654435761U; }

 protected:
  const int64_t topK_ = 10;
  const int64_t n_threads_ = 8;

 private:
  Profiler profiler_;
};

void ConcurrentTopKAggregateOp::run(Chunk &input_chunk, Chunk &res_chunk) {
  profiler_.Start();
  run_impl(input_chunk, res_chunk);
  profiler_.End();
  profiler_.Elapsed();
}

void ConcurrentTopKAggregateOp::convert_ht2topkheap(KEYCOUNT_HT &ht,
                                                    TOPK_HEAP &topk_heap) {
  if (ht.empty()) return;
  for (auto it = ht.begin(); it != ht.end(); ++it) {
    auto key = it->first;
    auto count = it->second;
    if (topk_heap.size() < topK_) {  // 不满topk个数据则直接插入
      topk_heap.push({key, count});
    } else if (topk_heap.top().count_ < count) {  // 当超过当前最小值是才插入
      topk_heap.push({key, count});
      topk_heap.pop();
    }
  }
  while (topk_heap.size() > topK_) {
    topk_heap.pop();
  }
}

void ConcurrentTopKAggregateOp::convert_heap2chunk(TOPK_HEAP &heap,
                                                   Chunk &chunk) {
  uint32_t agg_row_size = chunk.row_size_;

  std::vector<Item> topk_vector;
  while (!heap.empty()) {
    topk_vector.emplace_back(heap.top());
    heap.pop();
  }
  for (auto it = topk_vector.rbegin(); it != topk_vector.rend(); it++) {
    auto key = it->key_;
    auto count = it->count_;
    std::shared_ptr<Row> row(reinterpret_cast<Row *>(new char[agg_row_size]));
    row->cnt_ = 2;
    row->row_size_ = agg_row_size;
    Datum *key_datum = reinterpret_cast<Datum *>(row->payload_);
    Datum *count_datum = &key_datum[1];
    key_datum->int_ = key;
    key_datum->len_ = 8;
    count_datum->int_ = count;
    count_datum->len_ = 8;

    chunk.append_row(row.get());
  }
}

/**
 * =================================================================
 * Agg1 哈希表聚合方案
 * - 构建一个哈希表，将scan算子传入的数据依次插入哈希表完成聚合
 * - 挑选出count(*)最大的前topk个数据输出
 * =================================================================
 */
class Agg1Concurrent : public ConcurrentTopKAggregateOp {
 public:
  Agg1Concurrent(int64_t n_threads) : ConcurrentTopKAggregateOp(n_threads) {}

  void run_impl(Chunk &input_chunk, Chunk &res_chunk) override;

 private:
  struct mutex_wrapper : std::mutex {
    mutex_wrapper() = default;
    mutex_wrapper(mutex_wrapper const &) noexcept : std::mutex() {}
    bool operator==(mutex_wrapper const &other) noexcept {
      return this == &other;
    }
  };
  struct SafeMap {
    mutex_wrapper mtx_;
    KEYCOUNT_HT ht_;  // key, count(*)
  };
  std::vector<SafeMap> ht_;
  TOPK_HEAP topk_heap;
};

void Agg1Concurrent::run_impl(Chunk &input_chunk, Chunk &res_chunk) {
  int64_t n_partitions = 4096;
  int64_t mask = n_partitions - 1;
  ht_.resize(n_partitions);

  // 并发插入全局哈希表
  auto insert_ht = [&](int64_t thread_id, int64_t row_begin, int64_t row_end) {
    for (int i = row_begin; i < row_end; ++i) {
      Row *row = input_chunk.get_row(i);
      int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
      int64_t p = hash(key) & mask;
      {
        std::unique_lock<std::mutex> lock(ht_[p].mtx_);
        auto it = ht_[p].ht_.find(key);
        if (it == ht_[p].ht_.end()) {
          ht_[p].ht_[key] = 1;
        } else {
          it->second++;
        }
      }
    }
  };
  CONCURRENT_EXEC_FUNC(insert_ht, n_threads_, input_chunk.rows_);

  // 单线程遍历哈希表，求TopK
  for (int64_t i = 0; i < ht_.size(); ++i) {
    convert_ht2topkheap(ht_[i].ht_, topk_heap);
  }

  // 将堆转为chunk
  res_chunk.reserve(topk_heap.size() * res_chunk.row_size_);
  convert_heap2chunk(topk_heap, res_chunk);
}

/**
 * =================================================================
 * Agg2 分区哈希表聚合方案
 * - 将数据分区，依次聚合每个分区
 * - 每次聚合完一个分区，就更新一下全局TopK
 * =================================================================
 */
class Agg2Concurrent : public ConcurrentTopKAggregateOp {
  struct Partition {
    struct RowPointer {
      int64_t pos_;   // 行号
      int64_t hash_;  // key hash
    };
    std::vector<RowPointer> rows_;  // 属于该分区的row
  };

 public:
  Agg2Concurrent(int64_t n_threads) : ConcurrentTopKAggregateOp(n_threads) {}

  void run_impl(Chunk &input_chunk, Chunk &res_chunk) override;

 private:
  int64_t n_partitions_ = 4096;                       // 分区数
  std::vector<std::vector<Partition>> thread_parts_;  // 分区
  TOPK_HEAP heap_;  // 已经完成的聚合结果存在这里
};

void Agg2Concurrent::run_impl(Chunk &input_chunk, Chunk &res_chunk) {
  // 1. 数据分区，遍历所有row，将它们划分到不同分区
  int64_t mask = n_partitions_ - 1;  // 分区mask
  thread_parts_.resize(n_threads_, std::vector<Partition>(n_partitions_));
  auto partitioning_func = [&](int64_t thread_id, int64_t row_begin,
                               int64_t row_end) {
    auto &local_parts = thread_parts_[thread_id];
    for (int64_t i = row_begin; i < row_end; ++i) {
      Row *row = input_chunk.get_row(i);
      int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
      int64_t h = hash(key);
      int64_t p = h & mask;
      local_parts[p].rows_.push_back({i, h});
    }
  };
  CONCURRENT_EXEC_FUNC(partitioning_func, n_threads_, input_chunk.rows_);

  // 2. 聚合每个分区并求TopK
  std::vector<TOPK_HEAP> heaps(n_threads_);
  auto agg_func = [&](int64_t thread_id, int64_t row_begin, int64_t row_end) {
    KEYCOUNT_HT ht;
    auto &local_heap = heaps[thread_id];

    for (int64_t part_id = row_begin; part_id < row_end; ++part_id) {
      // 2.1 哈希聚合
      ht.clear();
      for (int64_t k = 0; k < n_threads_; ++k) {
        auto &part = thread_parts_[k][part_id];
        for (int row_id = 0; row_id < part.rows_.size(); ++row_id) {
          auto p = part.rows_[row_id].pos_;
          Row *row = input_chunk.get_row(p);
          int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
          auto it = ht.find(key);
          if (it != ht.end()) {
            it->second++;
          } else {
            ht[key] = 1;
          }
        }
      }

      // 2.2 更新TopK
      convert_ht2topkheap(ht, local_heap);
    }
  };
  CONCURRENT_EXEC_FUNC(agg_func, n_threads_, n_partitions_);

  // 3. 合并heap
  for (auto &local_heap : heaps) {
    while (!local_heap.empty()) {
      if (heap_.size() < topK_) {
        heap_.push(local_heap.top());
      } else if (heap_.top().count_ < local_heap.top().count_) {
        heap_.push(local_heap.top());
        heap_.pop();
      }
      local_heap.pop();
    }
  }

  // 3. 输出结果
  convert_heap2chunk(heap_, res_chunk);
}

/**
 * =================================================================
 * Agg3 面向TopK聚合的剪枝聚合算法
 *   1. 对数据进行partitioning多轮逻辑分区，并统计分区统计信息
 *   2. 挑选统计信息排名最高的分区作为候选分区，做物理聚合
 *   3. 用物理聚合的结果尝试剪枝逻辑分区
 *   4. 重复步骤1～3，直到剪枝率达到要求
 * =================================================================
 */
class Agg3Concurrent : public ConcurrentTopKAggregateOp {
  // 分区的状态
  enum PartitionState {
    FINISHED,  // 已经完成聚合或者已经被剪枝，不会参与后续的聚合和逻辑分区
    AGGing,    // 正在执行物理聚合
    LOGICAL    // 逻辑分区，只进行粗粒度统计信息收集
  };
  struct Partition {
    // 用于剪枝的统计信息，这里统计该分区的数据总行数
    // 注：这里只是demo，在真实系统实现时，统计信息有count，min，max，sum，ndv
    int64_t count_ = 0;

    // 分区状态，不同的分区状态对应的处理逻辑不同
    PartitionState state_ = PartitionState::FINISHED;

    // 用于做物理聚合的哈希表，只有在PartitionState::AGGing时才初始化
    std::unique_ptr<KEYCOUNT_HT> ht_;
  };

 public:
  Agg3Concurrent(int64_t n_threads) : ConcurrentTopKAggregateOp(n_threads) {}

  void run_impl(Chunk &input_chunk, Chunk &res_chunk) override;

 private:
  inline int64_t hash(int64_t key) { return key * 2654435761U; }

  /**
   * 根据上轮的分区信息，将数据重新进行逻辑分区
   * @param input_chunk 数据集
   * @param n_partitions 目标逻辑分区数
   */
  void partitioning(Chunk &input_chunk, int64_t n_partitions);

  /**
   * 在执行完一轮分区后，更新一下全局结果
   */
  void update_global_result();

  /**
   * 根据当前已经完成的部分聚合结果，得出TopK阈值，并尝试对逻辑分区进行剪枝
   */
  bool pruning();

  /**
   * 从分区中挑选出统计信息最靠前的几个分区，并标记其状态为下轮待物理聚合
   * @param n_candidate 要挑选标记的候选分区数量
   */
  void mark_candidate_partition(int64_t n_candidate);

  /**
   * 当不再进行逻辑分区后，对所有未被聚合的逻辑分区，执行物理聚合
   */
  void final_agg(Chunk &input_chunk);

 private:
  const int64_t MIN_PARTITIONS_ = 1 << 13;  // 8192
  const int64_t MAX_PARTITIONS_ = 1 << 18;  // 262144
  const int64_t REPARTITION_FACTOR = 8;     // 递归分区时扩大倍数
  const double PRUNING_THRESHOLD_ = 0.95;   // 95%
  std::vector<std::vector<Partition>> threads_partitions_;  // 线程局部分区
  std::vector<int64_t> global_stats_;  // 分区对应的全局统计结果
  TOPK_HEAP global_heap_;  // 已经完成的聚合结果存在这里
  int64_t round = 0;
};

void Agg3Concurrent::partitioning(Chunk &input_chunk,
                                  int64_t new_n_partitions) {
  // 上轮逻辑分区结果
  auto &old_threads_partitions = threads_partitions_;
  int64_t old_n_partitions = old_threads_partitions[0].size();
  int64_t old_mask = old_n_partitions - 1;
  // 本轮分区结果
  std::vector<std::vector<Partition>> new_threads_partitions(n_threads_);
  for (int64_t t = 0; t < n_threads_; ++t)
    new_threads_partitions[t].resize(new_n_partitions);
  int64_t new_mask = new_n_partitions - 1;

  // 分区：逻辑分区+物理聚合
  auto partitioning_func = [&](int64_t thread_id, int64_t row_begin,
                               int64_t row_end) {
    auto &old_partitions = old_threads_partitions[thread_id];
    auto &new_partitions = new_threads_partitions[thread_id];
    for (int64_t row_id = row_begin; row_id < row_end; ++row_id) {
      Row *row = input_chunk.get_row(row_id);
      int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
      int64_t h = hash(key);

      // 根据上轮的分区的状态，确定本轮分区的操作
      switch (old_partitions[h & old_mask].state_) {
        // 若上轮标记分区以及被聚合或被剪枝，则本轮属于该分区的row将不做任何处理
        case PartitionState::FINISHED: {
          continue;
        } break;

        // 若上轮标记分区为逻辑分区，则本轮对属于改逻辑分区的row做进一步逻辑划分
        case PartitionState::LOGICAL: {
          auto &partition = new_partitions[h & new_mask];
          partition.count_++;
          partition.state_ = PartitionState::LOGICAL;
        } break;

        // 若上轮标记分区需要做物理聚合，则本轮对属于改逻辑分区的row做哈希聚合
        case PartitionState::AGGing: {
          auto &partition = new_partitions[h & new_mask];
          if (partition.ht_.get() == nullptr) {
            partition.state_ = PartitionState::AGGing;
            partition.ht_ = std::unique_ptr<KEYCOUNT_HT>(new KEYCOUNT_HT());
          }
          auto it = partition.ht_->find(key);
          if (it != partition.ht_->end()) {
            it->second++;
          } else {
            (*partition.ht_)[key] = 1;
          }
        } break;

        default:
          break;
      }
    }
  };
  CONCURRENT_EXEC_FUNC(partitioning_func, n_threads_, input_chunk.rows_);

  // 替换为新的分区
  threads_partitions_ = std::move(new_threads_partitions);
}

void Agg3Concurrent::update_global_result() {
  int64_t n_partitions = threads_partitions_[0].size();

  // 每个线程收集的分区统计信息
  std::vector<int64_t> local_stats(n_partitions, 0);

  // 每个线程的TopK结果
  std::vector<TOPK_HEAP> local_heaps(n_threads_);

  auto traverse_partition_func = [&](int64_t thread_id, int64_t part_begin,
                                     int64_t part_end) {
    auto &local_heap = local_heaps[thread_id];
    // 遍历负责的分区
    for (int64_t part_id = part_begin; part_id < part_end; ++part_id) {
      // 用于收集每个线程的聚合结果
      KEYCOUNT_HT gather_ht;

      // 遍历每个线程的第part_id分区
      for (int64_t t = 0; t < n_threads_; ++t) {
        // 收集每个线程分区的统计信息到全局统计信息
        local_stats[part_id] += threads_partitions_[t][part_id].count_;

        // 判断这个分区是否是物理聚合分区，如果是，则将聚合结果插入到gather_ht中做最终聚合
        if (threads_partitions_[t][part_id].state_ != PartitionState::AGGing) {
          continue;
        }
        auto &partition = threads_partitions_[t][part_id];
        if (partition.ht_.get() == nullptr) {
          partition.state_ = PartitionState::FINISHED;
          continue;
        }
        auto &ht = *partition.ht_;
        if (gather_ht.empty()) {
          gather_ht = std::move(ht);
        } else {
          for (auto &[key, count] : ht) {
            auto it = gather_ht.find(key);
            if (it != gather_ht.end()) {
              it->second += count;
            } else {
              gather_ht[key] = count;
            }
          }
        }

        // 对于是AGGing的分区，并且已经将结果更新到全局TopK结果的分区，
        // 标记为处理完成，后续则不会对其再处理
        partition.state_ = PartitionState::FINISHED;
      }

      // 将最终聚合结果更新到全局TopK结果中
      for (auto &[key, count] : gather_ht) {
        if (local_heap.size() < topK_) {
          local_heap.push({key, count});
        } else if (local_heap.top().count_ < count) {
          local_heap.push({key, count});
          local_heap.pop();
        }
      }
    }
  };

  // 将每个线程的聚合结果合并
  CONCURRENT_EXEC_FUNC(traverse_partition_func, n_threads_, n_partitions);

  // 将此次分区中聚合的分区更新到全局结果中
  for (int64_t t = 0; t < n_threads_; ++t) {
    auto &local_heap = local_heaps[t];
    while (!local_heap.empty()) {
      auto &[key, count] = local_heap.top();
      if (global_heap_.size() < topK_) {
        global_heap_.push({key, count});
      } else if (global_heap_.top().count_ < count) {
        global_heap_.push({key, count});
        global_heap_.pop();
      }
      local_heap.pop();
    }
  }

  // 更新到全局
  global_stats_ = std::move(local_stats);
}

bool Agg3Concurrent::pruning() {
  int64_t n_partitions = threads_partitions_[0].size();

  if (global_heap_.size() >= topK_) {
    // 确保当前全局聚合结果只保留TopK个
    while (global_heap_.size() > topK_) global_heap_.pop();

    // 拿到当前已经完成聚合的阈值
    int64_t threshold = global_heap_.top().count_;

    // 遍历所有逻辑分区，若逻辑分区的粗粒度统计信息都小于当前聚合阈值，则分区被过滤
    std::vector<int64_t> threads_n_pruned(n_threads_, 0);
    auto pruning_func = [&](int64_t thread_id, int64_t part_begin,
                            int64_t part_end) {
      auto &local_n_pruned = threads_n_pruned[thread_id];
      auto &partitions = threads_partitions_[thread_id];
      for (int64_t part_id = part_begin; part_id < part_end; ++part_id) {
        auto &partition = partitions[part_id];
        if (partition.state_ == PartitionState::FINISHED) {
          local_n_pruned++;
        } else if (partition.state_ == PartitionState::LOGICAL &&
                   global_stats_[part_id] < threshold) {  // 剪枝成功
          local_n_pruned++;
          for (int64_t t = 0; t < n_threads_; ++t) {
            // 标记为被剪枝
            threads_partitions_[t][part_id].state_ = PartitionState::FINISHED;
          }
        }
      }
    };
    CONCURRENT_EXEC_FUNC(pruning_func, n_threads_, n_partitions);

    int64_t n_pruned = 0;
    for (auto &local_n_pruned : threads_n_pruned) {
      n_pruned += local_n_pruned;
    }
    double percent = n_pruned * 1.0 / n_partitions;
    printf("round: %ld, n_partitions: %ld, pruning percent: %f\n", round++,
           n_partitions, percent);

    // 若剪枝率达到要求，则停止逻辑分区
    if (percent >= PRUNING_THRESHOLD_) return true;
  } else {
    printf("round: %ld, n_partitions: %ld, pruning percent: %f\n", round++,
           n_partitions, 0.0);
  }
  return false;
}

void Agg3Concurrent::final_agg(Chunk &input_chunk) {
  int64_t n_partitions = threads_partitions_[0].size();
  int64_t mask = n_partitions - 1;

  // 每个线程聚合自己负责的剩余所有未聚合的分区
  auto agging_func = [&](int64_t thread_id, int64_t row_begin,
                         int64_t row_end) {
    auto &partitions = threads_partitions_[thread_id];
    for (int64_t row_id = row_begin; row_id < row_end; ++row_id) {
      Row *row = input_chunk.get_row(row_id);
      int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
      int64_t p = hash(key) & mask;
      auto &partition = partitions[p];
      if (partition.state_ == PartitionState::FINISHED) {
        continue;
      }

      if (partition.ht_.get() == nullptr) {
        partition.state_ = PartitionState::AGGing;
        partition.ht_ = std::unique_ptr<KEYCOUNT_HT>(new KEYCOUNT_HT());
      }
      auto it = partition.ht_->find(key);
      if (it != partition.ht_->end()) {
        it->second++;
      } else {
        (*partition.ht_)[key] = 1;
      }
    }
  };
  CONCURRENT_EXEC_FUNC(agging_func, n_threads_, input_chunk.rows_);

  // 合并线程局部的哈希表，并将最终聚合结果更新到全局topk结果中
  std::vector<TOPK_HEAP> local_heaps(n_threads_);  // 每个线程的TopK结果
  auto merge_ht_func = [&](int64_t thread_id, int64_t part_begin,
                           int64_t part_end) {
    auto &local_heap = local_heaps[thread_id];
    for (int64_t part_id = part_begin; part_id < part_end; ++part_id) {
      if (threads_partitions_[thread_id][part_id].state_ !=
          PartitionState::AGGing) {
        continue;
      }
      KEYCOUNT_HT gather_ht;
      for (int64_t t = 0; t < n_threads_; ++t) {
        auto &partition = threads_partitions_[t][part_id];
        if (partition.ht_.get() == nullptr) {
          partition.state_ = PartitionState::FINISHED;
          continue;
        }
        auto &ht = *partition.ht_;
        if (gather_ht.empty()) {
          gather_ht = std::move(ht);
        } else {
          for (auto &[key, count] : ht) {
            auto it = gather_ht.find(key);
            if (it != gather_ht.end()) {
              it->second += count;
            } else {
              gather_ht[key] = count;
            }
          }
        }

        // 对于是AGGing的分区，并且已经将结果更新到全局TopK结果的分区，
        // 标记为处理完成，后续则不会对其再处理
        partition.state_ = PartitionState::FINISHED;
      }

      for (auto &[key, count] : gather_ht) {
        if (local_heap.size() < topK_) {
          local_heap.push({key, count});
        } else if (local_heap.top().count_ < count) {
          local_heap.push({key, count});
          local_heap.pop();
        }
      }
    }
  };
  CONCURRENT_EXEC_FUNC(merge_ht_func, n_threads_, n_partitions);
  
  // 将此次分区中聚合的分区更新到全局结果中
  for (int64_t t = 0; t < n_threads_; ++t) {
    auto &local_heap = local_heaps[t];
    while (!local_heap.empty()) {
      auto &[key, count] = local_heap.top();
      if (global_heap_.size() < topK_) {
        global_heap_.push({key, count});
      } else if (global_heap_.top().count_ < count) {
        global_heap_.push({key, count});
        global_heap_.pop();
      }
      local_heap.pop();
    }
  }
}

void Agg3Concurrent::mark_candidate_partition(int64_t n_candidate) {
  int64_t n_partitions = global_stats_.size();

  // 筛选出统计信息最大的n_candidate个分区
  TOPK_HEAP candidate_heap;
  for (int64_t part_id = 0; part_id < n_partitions; ++part_id) {
    if (threads_partitions_[0][part_id].state_ == PartitionState::LOGICAL) {
      const auto &stat_count = global_stats_[part_id];
      if (candidate_heap.size() < n_candidate) {
        candidate_heap.push({part_id, stat_count});
      } else if (candidate_heap.top().count_ < stat_count) {
        candidate_heap.push({part_id, stat_count});
        candidate_heap.pop();
      }
    }
  }
  while (candidate_heap.size() > n_candidate) {
    candidate_heap.pop();
  }

  // 标记这n_candidate个分区，下轮执行物理分区
  while (!candidate_heap.empty()) {
    auto part_id = candidate_heap.top().key_;
    for (int64_t t = 0; t < n_threads_; ++t) {
      threads_partitions_[t][part_id].state_ = PartitionState::AGGing;
    }
    candidate_heap.pop();
  }
}

void Agg3Concurrent::run_impl(Chunk &input_chunk, Chunk &res_chunk) {
  threads_partitions_.resize(n_threads_);
  for (int64_t t = 0; t < n_threads_; ++t) {
    threads_partitions_[t].resize(1);
    threads_partitions_[t][0].state_ = PartitionState::LOGICAL;
  }

  int64_t n_partitions = MIN_PARTITIONS_;
  do {
    // 对数据进行逻辑分区
    partitioning(input_chunk, n_partitions);

    // 将新的分区的统计信息和部分聚合结果更新到全局
    update_global_result();

    // 分区剪枝，顺便判断是否满足剪枝阈值
    if (pruning()) break;

    // 标记候选分区，供下轮聚合
    mark_candidate_partition(topK_);

    // 确定新的分区数
    if (n_partitions == MAX_PARTITIONS_) {
      break;  // 达到分区上限，停止递归
    } else {
      n_partitions =
          std::min(MAX_PARTITIONS_, n_partitions * REPARTITION_FACTOR);
    }
  } while (1);

  // 最终物理聚合，将所有未聚合的逻辑分区进行物理聚合
  final_agg(input_chunk);

  convert_heap2chunk(global_heap_, res_chunk);
}
