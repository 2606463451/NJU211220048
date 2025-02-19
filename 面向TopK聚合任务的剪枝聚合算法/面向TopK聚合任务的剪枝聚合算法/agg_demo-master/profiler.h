#pragma once

#include <chrono>

//! The profiler can be used to measure elapsed time
class Profiler {
 public:
  Profiler() : finished(false), total(0) {}

  //! Starts the timer
  void Start() {
    finished = false;
    start = Tick();
  }
  //! Finishes timing
  void End() {
    end = Tick();
    finished = true;
  }

  //! Returns the elapsed time in seconds. If End() has been called, returns
  //! the total elapsed time. Otherwise returns how far along the timer is
  //! right now.
  double Elapsed() {
    auto _end = finished ? end : Tick();
    auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(_end - start)
            .count();
    total += duration;
    return duration;
  }

  double TotalTime() { return total; }

  void Reset() {
    finished = false;
    total = 0;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> Tick() const {
    return std::chrono::system_clock::now();
  }
  std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> end;
  bool finished = false;
  double total;
};
