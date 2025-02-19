#include <iostream>

#include "agg_op.hpp"
#include "profiler.h"
#include "scan_op.hpp"

template <class Agg>
void run(Chunk &input_chunk, std::string name) {
  printf("====================%s====================\n", name.c_str());
  Chunk res_chunk(sizeof(Row) + sizeof(Datum) * 2);
  std::shared_ptr<Agg> agg_op(new Agg());
  agg_op->run(input_chunk, res_chunk);
  printf("%s Time: %fs\n", name.c_str(), agg_op->get_runtime());
  printf("--------------------------------------------\n");
  printf("[group, count(*)]\n");
  for (int i = 0; i < res_chunk.rows_; ++i) {
    Row *row = res_chunk.get_row(i);
    auto s = row->to_string();
    printf("%s\n", s.c_str());
  }
  printf("============================================\n\n");
}

int main() {
  Profiler profiler;

  // 加载数据集
  Chunk input_chunk(sizeof(Row) + sizeof(Datum) * 1);
  int64_t n_rows = 100000000;
  profiler.Start();
  {
    std::string path = "./data/userid_100M";
    std::shared_ptr<ScanOp> scan_op(new ScanOp(path));
    scan_op->open();
    scan_op->scan_all(input_chunk, n_rows, sizeof(int64_t));
  }
  profiler.End();
  printf("Load Data Time: %fs\n", profiler.Elapsed());

  run<Agg1>(input_chunk, "Agg1");
  run<Agg2>(input_chunk, "Agg2");
  run<Agg3>(input_chunk, "Agg3");
}
