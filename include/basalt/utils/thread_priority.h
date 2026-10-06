#pragma once

#ifdef __linux__
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace basalt {

// Raises the calling thread's nice value so latency-tolerant background work
// yields to tracking when the CPU is saturated. No-op off Linux.
inline void lowerCurrentThreadPriority(int nice_value = 10) {
#ifdef __linux__
  setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), nice_value);
#else
  (void)nice_value;
#endif
}

}  // namespace basalt
