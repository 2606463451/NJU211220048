# Makefile

# 指定编译器
CXX = g++

# 编译器选项
CXXFLAGS = -std=c++17 -O2

# 链接库
LDFLAGS = -lpthread

# 源文件
SRC1 = single_thread_test.cpp
SRC2 = multi_thread_test.cpp

# 目标文件
OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cpp=.o)

# 可执行文件
TARGET1 = single_thread_test.out
TARGET2 = multi_thread_test.out

# 默认目标
all: $(TARGET1) $(TARGET2)

# 目标文件规则
$(OBJ1): $(SRC1)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ2): $(SRC2)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 可执行文件规则
$(TARGET1): $(OBJ1)
	$(CXX) $(OBJ1) $(LDFLAGS) -o $@

$(TARGET2): $(OBJ2)
	$(CXX) $(OBJ2) $(LDFLAGS) -o $@

# 清理生成文件
clean:
	rm -f $(OBJ1) $(OBJ2) $(TARGET1) $(TARGET2)

# 伪目标
.PHONY: all clean
