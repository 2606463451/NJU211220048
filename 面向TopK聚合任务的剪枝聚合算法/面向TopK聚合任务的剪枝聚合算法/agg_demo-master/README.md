# TopK Aggregation Demo

## build
```bash
# 1. 创建数据集文件夹
mkdir data/

# 2. 拷贝数据集到data/中
# xxx

# 3. 编译
make
```

## run
```bash
./single_thread_test.out  # 运行单线程测试
./multi_thread_test.out 4 # 运行4线程测试
```

## code
- 如果更换数据集，需要在single_thread_test.cpp和multi_thread_test.cpp中更新n_rows和path的值
- n_rows代表数据集中数据数量，path表示数据集路径
