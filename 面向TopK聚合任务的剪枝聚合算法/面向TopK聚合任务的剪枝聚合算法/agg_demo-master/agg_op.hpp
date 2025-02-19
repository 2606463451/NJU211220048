#pragma once
#include <algorithm>
#include <memory>
#include <queue>
#include <unordered_map>

#include "profiler.h"
#include "row.hpp"

/**
 * TopKAgg算子的基类
 * - 只实现了group by(int64_t) + count(*) + order by(count(*)) + limit的功能
 */
class TopKAggregateOp {
 public:
  TopKAggregateOp() {}

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

 protected:
  static const int64_t topK_ = 10;

 private:
  Profiler profiler_;
};

void TopKAggregateOp::run(Chunk &input_chunk, Chunk &res_chunk) {
  profiler_.Start();
  run_impl(input_chunk, res_chunk);
  profiler_.End();
  profiler_.Elapsed();
}

void TopKAggregateOp::convert_ht2topkheap(KEYCOUNT_HT &ht,
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

void TopKAggregateOp::convert_heap2chunk(TOPK_HEAP &heap, Chunk &chunk) {
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
class Agg1 : public TopKAggregateOp {
 public:
  Agg1() : TopKAggregateOp() {}

  void run_impl(Chunk &input_chunk, Chunk &res_chunk) override;

 private:
  KEYCOUNT_HT ht_;  // key, count(*)
};

void Agg1::run_impl(Chunk &input_chunk, Chunk &res_chunk) {
  // 插入全局哈希表
  for (int i = 0; i < input_chunk.rows_; ++i) {
    Row *row = input_chunk.get_row(i);
    int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
    auto it = ht_.find(key);
    if (it == ht_.end()) {
      ht_[key] = 1;
    } else {
      it->second++;
    }
  }

  // 求TopK
  TOPK_HEAP topk_heap;
  convert_ht2topkheap(ht_, topk_heap);

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
class Agg2 : public TopKAggregateOp {
  struct Partition {
    struct RowPointer {
      int64_t pos_;   // 行号
      int64_t hash_;  // key hash
    };
    std::vector<RowPointer> rows_;  // 属于该分区的row
  };

 public:
  Agg2() : TopKAggregateOp() {}

  void run_impl(Chunk &input_chunk, Chunk &res_chunk) override;

 private:
  inline int64_t hash(int64_t key) { return key * 2654435761U; }

 private:
  int64_t n_partitions_ = 4096;   // 分区数
  std::vector<Partition> parts_;  // 分区
  TOPK_HEAP heap_;                // 已经完成的聚合结果存在这里
};

void Agg2::run_impl(Chunk &input_chunk, Chunk &res_chunk) {
  // 1. 数据分区，遍历所有row，将它们划分到不同分区
  parts_.resize(n_partitions_);
  int64_t mask = n_partitions_ - 1;  // 分区mask
  for (int64_t i = 0; i < input_chunk.rows_; ++i) {
    Row *row = input_chunk.get_row(i);
    int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
    int64_t h = hash(key);
    int64_t p = h & mask;
    parts_[p].rows_.push_back({i, h});
  }

  // // 2. 聚合每个分区并求TopK
  KEYCOUNT_HT ht;
  for (int64_t part_id = 0; part_id < n_partitions_; ++part_id) {
    auto &part = parts_[part_id];

    // 2.1 哈希聚合
    ht.clear();
    for (int64_t row_id = 0; row_id < part.rows_.size(); ++row_id) {
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

    // 2.2 更新TopK
    convert_ht2topkheap(ht, heap_);
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
class Agg3 : public TopKAggregateOp {
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
  Agg3() : TopKAggregateOp() {}

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
   * 根据当前已经完成的部分聚合结果，得出TopK阈值，并尝试对逻辑分区进行剪枝
   */
  bool pruning();

  /**
   * 从分区中挑选出统计信息最靠前的几个分区，并标记其状态为下轮待物理聚合
   * @param partitions 所有分区
   * @param n_candidate 要挑选标记的候选分区数量
   */
  static void mark_candidate_partition(std::vector<Partition> &partitions,
                                       int64_t n_candidate);

  /**
   * 当不再进行逻辑分区后，对所有未被聚合的逻辑分区，执行物理聚合
   */
  void final_agg(Chunk &input_chunk);

 private:
  const int64_t MIN_PARTITIONS_ = 1 << 13;  // 8192
  const int64_t MAX_PARTITIONS_ = 1 << 18;  // 262144
  const int64_t REPARTITION_FACTOR = 8;     // 递归分区时扩大倍数
  const double PRUNING_THRESHOLD_ = 0.95;   // 95%
  std::vector<Partition> partitions_;       // 分区
  TOPK_HEAP global_heap_;  // 已经完成的聚合结果存在这里
  int64_t round = 0;
};

void Agg3::partitioning(Chunk &input_chunk, int64_t new_n_partitions) {
  // 上轮逻辑分区结果
  auto &old_partitions = partitions_;
  int64_t old_n_partitions = old_partitions.size();
  int64_t old_mask = old_n_partitions - 1;
  // 本轮分区结果
  std::vector<Partition> new_partitions(new_n_partitions);
  int64_t new_mask = new_n_partitions - 1;

  // 分区：逻辑分区+物理聚合
  for (int64_t row_id = 0; row_id < input_chunk.rows_; ++row_id) {
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
  // 替换为新的分区
  partitions_ = std::move(new_partitions);

  // 将此次分区中聚合的分区更新到全局结果中
  for (int64_t part_id = 0; part_id < partitions_.size(); ++part_id) {
    auto &partition = partitions_[part_id];
    if (partition.state_ != PartitionState::AGGing) {
      continue;
    }

    auto &ht = *partition.ht_;
    for (auto &[key, count] : ht) {
      if (global_heap_.size() < topK_) {
        global_heap_.push({key, count});
      } else if (global_heap_.top().count_ < count) {
        global_heap_.push({key, count});
        global_heap_.pop();
      }
    }

    partition.state_ = PartitionState::FINISHED;
  }
}

bool Agg3::pruning() {
  if (global_heap_.size() >= topK_) {
    // 确保当前全局聚合结果只保留TopK个
    while (global_heap_.size() > topK_) global_heap_.pop();

    // 拿到当前已经完成聚合的阈值
    int64_t threshold = global_heap_.top().count_;

    // 遍历所有逻辑分区，若逻辑分区的粗粒度统计信息都小于当前聚合阈值，则分区被过滤
    int64_t n_pruned = 0;
    for (int64_t part_id = 0; part_id < partitions_.size(); ++part_id) {
      auto &partition = partitions_[part_id];
      if (partition.state_ == PartitionState::FINISHED) {
        n_pruned++;
      } else if (partition.state_ == PartitionState::LOGICAL &&
                 partition.count_ < threshold) {  // 剪枝成功
        n_pruned++;
        partition.state_ = PartitionState::FINISHED;  // 标记为被剪枝
      }
    }
    double percent = n_pruned * 1.0 / partitions_.size();
    printf("round: %ld, n_partitions: %ld, pruning percent: %f\n", round++,
           partitions_.size(), percent);

    // 若剪枝率达到要求，则停止逻辑分区
    if (percent >= PRUNING_THRESHOLD_) return true;
  } else {
    printf("round: %ld, n_partitions: %ld, pruning percent: %f\n", round++,
           partitions_.size(), 0.0);
  }
  return false;
}

void Agg3::mark_candidate_partition(std::vector<Partition> &partitions,
                                    int64_t n_candidate) {
  TOPK_HEAP candidate_heap;
  for (int64_t part_id = 0; part_id < partitions.size(); ++part_id) {
    if (partitions[part_id].state_ == PartitionState::LOGICAL) {
      if (candidate_heap.size() < n_candidate) {
        candidate_heap.push({part_id, partitions[part_id].count_});
      } else if (candidate_heap.top().count_ < partitions[part_id].count_) {
        candidate_heap.push({part_id, partitions[part_id].count_});
        candidate_heap.pop();
      }
    }
  }
  while (candidate_heap.size() > n_candidate) {
    candidate_heap.pop();
  }

  while (!candidate_heap.empty()) {
    auto part_id = candidate_heap.top().key_;
    partitions[part_id].state_ = PartitionState::AGGing;
    candidate_heap.pop();
  }
}

void Agg3::final_agg(Chunk &input_chunk) {
  int mask = partitions_.size() - 1;

  // 聚合剩余所有未聚合的分区
  for (int64_t row_id = 0; row_id < input_chunk.rows_; ++row_id) {
    Row *row = input_chunk.get_row(row_id);
    int64_t key = reinterpret_cast<Datum *>(row->payload_)->int_;
    int64_t p = hash(key) & mask;
    auto &partition = partitions_[p];
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

  // 更新全局topk结果
  for (int64_t part_id = 0; part_id < partitions_.size(); ++part_id) {
    auto &partition = partitions_[part_id];
    if (partition.state_ != PartitionState::AGGing) {
      continue;
    }

    auto &ht = *partition.ht_;
    for (auto &[key, count] : ht) {
      if (global_heap_.size() < topK_) {
        global_heap_.push({key, count});
      } else if (global_heap_.top().count_ < count) {
        global_heap_.push({key, count});
        global_heap_.pop();
      }
    }

    partition.state_ = PartitionState::FINISHED;
  }
}

void Agg3::run_impl(Chunk &input_chunk, Chunk &res_chunk) {
  int64_t n_partitions = MIN_PARTITIONS_;
  partitions_.resize(1);
  partitions_[0].state_ = PartitionState::LOGICAL;
  do {
    // 对数据进行逻辑分区
    partitioning(input_chunk, n_partitions);

    // 分区剪枝，顺便判断是否满足剪枝阈值
    if (pruning()) break;

    // 标记候选分区，供下轮聚合
    mark_candidate_partition(partitions_, topK_);

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
