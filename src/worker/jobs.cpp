#include "kmm/worker/jobs.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

struct ExecuteJob::Result: ITaskCompletion {
    Result(std::shared_ptr<const Waker> job) : m_job(std::move(job)) {}
    Result(TaskResult result) : m_inner(std::move(result)) {}

    void complete_task(TaskResult result) final {
        if (auto job = std::exchange(this->m_job, nullptr)) {
            m_inner = std::move(result);
            job->trigger_wakeup(true);
        }
    }

    std::optional<TaskResult> take_result() {
        if (!m_inner) {
            return std::nullopt;
        }

        m_job = nullptr;
        return std::move(m_inner);
    }

  private:
    std::shared_ptr<const Waker> m_job;
    std::optional<TaskResult> m_inner;
};

PollResult ExecuteJob::poll(WorkerState& worker) {
    if (m_status == Status::Created) {
        auto requests = std::vector<MemoryRequest>();

        for (const auto& arg : m_inputs) {
            auto buffer_id_opt = worker.block_manager.get_block_buffer(arg.block_id);

            if (buffer_id_opt) {
                requests.emplace_back(worker.memory_manager->create_request(  //
                    *buffer_id_opt,
                    arg.memory_id,
                    AccessMode::Read,
                    shared_from_this()));
            } else {
                requests.emplace_back(nullptr);
            }
        }

        for (const auto& arg : m_outputs) {
            auto layout = arg.header->layout();

            if (layout.num_bytes > 0) {
                auto buffer_id = worker.memory_manager->create_buffer(arg.header->layout());

                m_output_buffers.emplace_back(buffer_id);
                requests.emplace_back(worker.memory_manager->create_request(  //
                    buffer_id,
                    arg.memory_id,
                    AccessMode::ReadWrite,
                    shared_from_this()));
            } else {
                m_output_buffers.emplace_back(std::nullopt);
                requests.emplace_back(nullptr);
            }
        }

        m_status = Status::Staging;
        m_memory_requests = std::move(requests);
    }

    if (m_status == Status::Staging) {
        if (worker.memory_manager->poll_requests(m_memory_requests) != PollResult::Ready) {
            return PollResult::Pending;
        }

        try {
            auto result = std::make_shared<Result>(shared_from_this());
            auto context = TaskContext(TaskCompletion(result));

            unsigned long index = 0;

            for (const auto& input : m_inputs) {
                const auto& req = m_memory_requests[index++];
                auto block = worker.block_manager.get_block(input.block_id);
                const auto* allocation = req ? worker.memory_manager->view_buffer(req) : nullptr;

                context.inputs.push_back(BlockAccessor {
                    .block_id = input.block_id,
                    .header = block.header,
                    .allocation = allocation,
                });
            }

            size_t output_index = 0;

            for (const auto& output : m_outputs) {
                const auto& req = m_memory_requests[index++];
                const auto* allocation = req ? worker.memory_manager->view_buffer(req) : nullptr;

                context.outputs.push_back(BlockAccessorMut {
                    .block_id = BlockId(m_id, static_cast<uint8_t>(output_index++)),
                    .header = output.header.get(),
                    .allocation = allocation,
                });
            }

            m_result = std::move(result);
            worker.executors.at(m_device_id)->submit(m_task, std::move(context));
        } catch (const std::exception& e) {
            m_result = std::make_shared<Result>(TaskError(e));
        }

        m_status = Status::Running;
    }

    if (m_status == Status::Running) {
        auto result = m_result->take_result();

        if (!result.has_value()) {
            return PollResult::Pending;
        }

        for (const auto& request : m_memory_requests) {
            if (request) {
                worker.memory_manager->delete_request(request);
            }
        }

        m_memory_requests.clear();

        unsigned long num_outputs = m_outputs.size();
        const auto* error = std::get_if<TaskError>(&*result);

        for (unsigned long i = 0; i < num_outputs; i++) {
            auto& output = m_outputs[i];
            auto block_id = BlockId(m_id, static_cast<uint8_t>(i));
            auto buffer_id = m_output_buffers[i];

            if (error == nullptr) {
                worker.block_manager.insert_block(  //
                    block_id,
                    std::move(output.header),
                    output.memory_id,
                    buffer_id);
            } else {
                worker.block_manager.poison_block(block_id, *error);

                if (buffer_id) {
                    worker.memory_manager->delete_buffer(*buffer_id);
                }
            }
        }

        m_status = Status::Done;
    }

    return PollResult::Ready;
}

PollResult DeleteJob::poll(WorkerState& worker) {
    if (m_block_id) {
        spdlog::debug("delete block id={}", *m_block_id);
        m_buffer_id = worker.block_manager.delete_block(*m_block_id);
        m_block_id = std::nullopt;

        if (m_buffer_id) {
            worker.memory_manager->decrement_buffer_refcount(*m_buffer_id);
        }
    }

    return PollResult::Ready;
}

PollResult PrefetchJob::poll(WorkerState& state) {
    if (m_status == Status::Created) {
        std::optional<BufferId> buffer_id = state.block_manager.get_block_buffer(m_block_id);

        // Not all blocks have an associated buffer
        if (!buffer_id.has_value()) {
            m_status = Status::Done;
            return PollResult::Ready;
        }

        m_memory_request = state.memory_manager->create_request(  //
            *buffer_id,
            m_memory_id,
            AccessMode::Read,
            shared_from_this());

        m_status = Status::Active;
    }

    if (m_status == Status::Active) {
        if (state.memory_manager->poll_request(m_memory_request) != PollResult::Ready) {
            return PollResult::Pending;
        }

        state.memory_manager->delete_request(m_memory_request);
        m_memory_request = nullptr;

        m_status = Status::Done;
    }

    return PollResult::Ready;
}

PollResult EmptyJob::poll(WorkerState& worker) {
    return PollResult::Ready;
}

std::shared_ptr<Job> build_job_for_command(EventId id, Command command) {
    if (auto* cmd_exe = std::get_if<ExecuteCommand>(&command)) {
        return std::make_shared<ExecuteJob>(id, std::move(*cmd_exe));
    } else if (auto* cmd_del = std::get_if<BlockDeleteCommand>(&command)) {
        return std::make_shared<DeleteJob>(id, std::move(*cmd_del));
    } else if (const auto* cmd_noop = std::get_if<EmptyCommand>(&command)) {
        return std::make_shared<EmptyJob>(id);
    } else if (auto* cmd_fetch = std::get_if<BlockPrefetchCommand>(&command)) {
        return std::make_shared<PrefetchJob>(id, std::move(*cmd_fetch));
    } else {
        KMM_PANIC("invalid command");
    }
}

}  // namespace kmm