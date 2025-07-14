#include "spdlog/spdlog.h"

#include "kmm/runtime/executor.hpp"

namespace kmm {

Executor::Executor(
    std::shared_ptr<DeviceResources> device_resources,
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::shared_ptr<BufferRegistry> buffer_registry,
    std::shared_ptr<Scheduler> scheduler,
    bool debug_mode
) :
    m_device_resources(device_resources),
    m_stream_manager(stream_manager),
    m_buffer_registry(buffer_registry),
    m_scheduler(scheduler),
    m_debug_mode(debug_mode) {}

Executor::~Executor() {}

bool Executor::is_idle() const {
    return m_jobs_head == nullptr;
}

void Executor::make_progress() {
    Job* prev = nullptr;
    std::unique_ptr<Job>* current_ptr = &m_jobs_head;

    while (auto* current = current_ptr->get()) {
        // In debug mode, only poll the head task (prev == nullptr)
        bool should_poll = !m_debug_mode || prev == nullptr;

        if (should_poll && current->poll(*this) == Poll::Ready) {
            *current_ptr = std::move(current->next);
        } else {
            prev = current;
            current_ptr = &current->next;
        }
    }

    m_jobs_tail = prev;
}

void Executor::execute_task(TaskHandle task, DeviceEventSet dependencies) {
    const Command& command = task->get_command();

    // For the case of an empty task, we provide a fast path if all device events are done
    if (m_stream_manager->is_ready(dependencies) && std::holds_alternative<CommandEmpty>(command)) {
        m_scheduler->mark_as_completed(task);
        return;
    }

    auto job = build_job_for_command(task, command, dependencies);

    if (auto* old_tail = std::exchange(m_jobs_tail, job.get())) {
        old_tail->next = std::move(job);
    } else {
        m_jobs_head = std::move(job);
    }
}

}  // namespace kmm
