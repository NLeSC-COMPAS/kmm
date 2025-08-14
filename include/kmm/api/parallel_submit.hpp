#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/api/task_group.hpp"
#include "kmm/core/buffer.hpp"
#include "kmm/core/domain.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

class Runtime;
class TaskGraphState;

namespace detail {

template<typename Launcher, typename... Args>
class ComputeTaskImpl: public ComputeTask {
  public:
    ComputeTaskImpl(DomainChunk chunk, Launcher launcher, Args... args) :
        m_chunk(chunk),
        m_launcher(std::move(launcher)),
        m_args(std::move(args)...) {}

    void execute(Resource& resource, TaskContext context) override {
        execute_impl(std::index_sequence_for<Args...>(), resource, context);
    }

    template<size_t... Is>
    void execute_impl(std::index_sequence<Is...>, Resource& resource, TaskContext& context) {
        static constexpr ExecutionSpace execution_space = Launcher::execution_space;

        m_launcher(
            resource,
            m_chunk,
            ArgumentUnpack<execution_space, Args>::call(context, std::get<Is>(m_args))...
        );
    }

  private:
    DomainChunk m_chunk;
    Launcher m_launcher;
    std::tuple<Args...> m_args;
};

template<size_t... Is, typename Launcher, typename... Args>
EventId parallel_submit_impl(
    std::index_sequence<Is...>,
    Runtime& runtime,
    const SystemInfo& system_info,
    const Domain& domain,
    Launcher launcher,
    Args&&... args
) {
    std::tuple<ArgumentHandler<Args>...> handlers = {std::forward<Args>(args)...};

    auto init = TaskGroupInit {
        .runtime = runtime,  //
        .domain = domain
    };

    (std::get<Is>(handlers).initialize(init), ...);

    return runtime.schedule([&](TaskGraph& graph) {
        EventList events;

        for (const DomainChunk& chunk : domain.chunks) {
            auto processor_id = chunk.owner_id;

            auto instance = TaskInstance {
                .runtime = runtime,
                .graph = graph,
                .chunk = chunk,
                .memory_id = system_info.affinity_memory(processor_id),
                .buffers = {},
                .dependencies = {}
            };

            auto task = std::make_unique<ComputeTaskImpl<Launcher, packed_argument_t<Args>...>>(
                chunk,
                launcher,
                std::get<Is>(handlers).before_submit(instance)...
            );

            EventId event_id = graph.insert_compute_task(
                processor_id,
                std::move(task),
                std::move(instance.buffers),
                std::move(instance.dependencies)
            );

            events.push_back(event_id);

            auto result = TaskSubmissionResult {
                .runtime = runtime,  //
                .graph = graph,
                .event_id = event_id
            };

            (std::get<Is>(handlers).after_submit(result), ...);
        }

        auto commit = TaskGroupCommit {.runtime = runtime, .graph = graph};

        (std::get<Is>(handlers).commit(commit), ...);

        return graph.join_events(events);
    });
}
}  // namespace detail

template<typename Launcher, typename... Args>
EventId parallel_submit(
    Runtime& runtime,
    const SystemInfo& system_info,
    const Domain& partition,
    Launcher launcher,
    Args&&... args
) {
    return detail::parallel_submit_impl(
        std::index_sequence_for<Args...> {},
        runtime,
        system_info,
        partition,
        launcher,
        std::forward<Args>(args)...
    );
}

}  // namespace kmm
