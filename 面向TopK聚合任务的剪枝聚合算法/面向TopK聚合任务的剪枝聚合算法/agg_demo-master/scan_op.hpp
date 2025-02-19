#pragma once
#include <memory>

#include "file_system.hpp"
#include "row.hpp"

class ScanOp {
 public:
  ScanOp(const std::string &path);
  ~ScanOp();
  void open();
  void scan_all(Chunk &chunk, int64_t n_rows, uint32_t item_size);

 private:
  const std::string file_name_;
  int64_t file_fd_;  // 要读取的数据文件
  bool is_open_;
  bool is_end_;
};

ScanOp::ScanOp(const std::string &path)
    : file_name_(path), file_fd_(-1), is_open_(false), is_end_(false) {}

ScanOp::~ScanOp() {
  FileSystem::CloseFile(file_fd_);
  is_open_ = false;
  is_end_ = false;
}

void ScanOp::open() {
  file_fd_ = FileSystem::OpenFile(file_name_.c_str());
  is_open_ = file_fd_ != -1;
}

void ScanOp::scan_all(Chunk &chunk, int64_t n_rows, uint32_t item_size) {
  if (!is_open_ || is_end_) return;

  //===================================
  //   当前默认文件格式是单列int64_t数据
  //===================================
  
  // 从文件读取数据
  int64_t n_bytes = n_rows * item_size;
  char *buf = (char *)malloc(n_bytes);
  FileSystem::ReadFile(file_fd_, buf, n_bytes);

  // 构造row并写入chunk
  uint32_t row_size = sizeof(Row) + sizeof(Datum) * 1;
  chunk.reserve(row_size * n_rows);
  for (int64_t i = 0; i < n_rows; ++i) {
    std::shared_ptr<Row> row(reinterpret_cast<Row *>(new char[row_size]));
    row->cnt_ = 1;
    row->row_size_ = row_size;
    Datum *datum = reinterpret_cast<Datum *>(row->payload_);
    datum->int_ = *reinterpret_cast<int64_t *>(buf + i * sizeof(int64_t));
    datum->len_ = 8;
    
    chunk.append_row(row.get());
  }

  is_end_ = true;
}
