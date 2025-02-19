#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

namespace FileSystem {

int OpenFile(const char *path) { return open(path, O_RDWR | O_CREAT, 0666); }

// Read n_bytes into buf from fd. Return the
// number read, -1 for errors or 0 for EOF.
int ReadFile(int fd, char *buf, int64_t n_bytes) {
  return read(fd, buf, n_bytes);
}

bool ReadLine(int fd, std::string &line) {
  line.clear();
  char buffer[1];
  while (true) {
    int64_t tuples_read = ReadFile(fd, buffer, 1);
    if (tuples_read == 0 || buffer[0] == '\n') {
      return !line.empty();
    }
    if (buffer[0] != '\r') {
      line += buffer[0];
    }
  }
}

int WriteFile(int fd, char *buf, int64_t n_bytes) {
  return write(fd, buf, n_bytes);
}

void CloseFile(int fd) {
  if (fd != -1) {
    close(fd);
  }
}

};  // namespace FileSystem