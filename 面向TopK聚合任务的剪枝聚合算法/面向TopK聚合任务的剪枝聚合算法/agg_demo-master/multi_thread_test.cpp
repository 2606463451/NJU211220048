#include "scan_op.hpp"
#include "agg_op_concurrent.hpp"
#include "profiler.h"
#include <iostream>

int64_t n_threads = 4;

template <class Agg>
void run(Chunk &input_chunk, std::string name) {
  printf("====================%s====================\n", name.c_str());
  Chunk res_chunk(sizeof(Row) + sizeof(Datum) * 2);
  std::shared_ptr<Agg> agg_op(new Agg(n_threads));
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

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Please input thread number, i.e. multi_thread_test 4\n");
    return 0;
  }

  n_threads = std::atoi(argv[1]);

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

  printf("n_threads: %ld\n", n_threads);
  
  run<Agg1Concurrent>(input_chunk, "Agg1Concurrent");
  run<Agg2Concurrent>(input_chunk, "Agg2Concurrent");
  run<Agg3Concurrent>(input_chunk, "Agg3Concurrent");
}

