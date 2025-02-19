#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <string.h>

/**
 * 表示一个数据
 * - 如果是一个整数或浮点数，则数据就实际存储在datum
 * - 如果是一个字符串，则datum只记录的指向字符串的指针
 */
struct Datum {
  union {
    char *ptr_;
    int64_t int_;
    uint64_t uint_;
    double double_;
  };

  union {
    struct {
      uint32_t len_ : 29;
      uint32_t flag_ : 2;
      uint32_t null_ : 1;
    };
    uint32_t pack_;
  };

} __attribute__((packed));


/**
 * 表示一行数据
 * - 格式: | Datum1, Datum2, ... | d1, d2, ... |
 * - row的开始是连续数个Datum，表示row的几个属性，如果有变长属性，则连续的存储在row后面
 * - 当前只允许存储一个int64_t的数据
 */
struct Row {
 public:
  inline const Datum *CellAt(int64_t p) const;
  static void deep_copy(Row *l, const Row *r);
  std::string to_string();

  uint32_t cnt_;      // row中有多少列数据
  uint32_t row_size_; // 整个Row的大小，包含cnt_, row_size_
  char payload_[0];   // 前半部分是cnt_个datum，后半部分存储变长属性值
} __attribute__((packed));

/**
 * 存储多行Row
 * - 当前只支持定长row
 */
struct Chunk {
public:
  Chunk(uint32_t row_size) : rows_(0), used_pos_(0), row_size_(row_size) {}
  void reserve(int64_t size);
  void append_row(const Row *row);

  // 定长row的获取
  Row *get_row(int64_t row_pos);

  uint32_t rows_;      /* number of rows */
  uint32_t row_size_;  /* fixed row size */
  uint32_t used_pos_;  /* current used postion */
  std::vector<char> payload_;
};

void Row::deep_copy(Row *l, const Row *r) {
  l->cnt_ = r->cnt_;
  l->row_size_ = r->row_size_;
  memcpy(l->payload_, r->payload_, r->row_size_ - sizeof(Row));
}

std::string Row::to_string() {
  std::string s = "[";
  auto data = reinterpret_cast<Datum *>(payload_);
  for (int i = 0; i < cnt_; ++i) {
    s += std::to_string(data[i].int_);
    if (i < cnt_ - 1) s += ", ";
  }
  s += "]";

  return std::move(s);
}

void Chunk::reserve(int64_t size) {
  if (size <= payload_.size()) return;
  payload_.resize(size);
}

void Chunk::append_row(const Row *row) {
  int64_t now_size = payload_.size() > 0 ? payload_.size() : 1;
  while (used_pos_ + row->row_size_ > now_size) now_size *= 2;
  if (now_size > payload_.size()) payload_.resize(now_size);

  Row::deep_copy(reinterpret_cast<Row *>(&payload_[used_pos_]), row);
  used_pos_ += row->row_size_;
  rows_++;
}

Row *Chunk::get_row(int64_t row_pos) {
  if (row_pos >= rows_) return nullptr;
  return reinterpret_cast<Row*>(&payload_[row_pos * row_size_]);
}
