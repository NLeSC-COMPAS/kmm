#pragma once

#include "memory_system.hpp"
#include "scheduler.hpp"

namespace kmm {

struct WorkerState {
    Scheduler& scheduler() {
        return m_scheduler;
    }

  protected:
    Scheduler m_scheduler;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<MemorySystem> m_memory_system;
};

}  // namespace kmm