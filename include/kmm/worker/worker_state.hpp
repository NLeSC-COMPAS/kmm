#pragma once

#include "buffer_registry.hpp"
#include "memory_manager.hpp"
#include "memory_system.hpp"
#include "scheduler.hpp"

namespace kmm {

struct WorkerState {
    WorkerState(
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<MemorySystem> memory_system,
        std::shared_ptr<Scheduler> scheduler
    ) :
        m_memory_system(memory_system),
        m_memory_manager(std::make_shared<MemoryManager>(memory_system)),
        m_buffer_registry(std::make_shared<BufferRegistry>(m_memory_manager)),
        m_stream_manager(stream_manager),
        m_scheduler(scheduler) {}

  protected:
    std::shared_ptr<MemorySystem> m_memory_system;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<BufferRegistry> m_buffer_registry;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::shared_ptr<Scheduler> m_scheduler;
};

}  // namespace kmm