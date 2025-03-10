#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/api/task_group.hpp"
#include "kmm/core/buffer.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/dag/domain.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

class Worker;
class TaskGraph;

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
    Worker& worker,
    const SystemInfo& system_info,
    const Domain& domain,
    Launcher launcher,
    Args&&... args
) {
    std::tuple<ArgumentHandler<Args>...> handlers = {
        ArgumentHandler<Args>(std::forward<Args>(args))...};

    auto init = TaskGroupInit {
        .worker = worker,  //
        .domain = domain};

    (std::get<Is>(handlers).initialize(init), ...);

    return worker.with_task_graph([&](TaskGraph& graph) {
        EventList events;

        for (const DomainChunk& chunk : domain.chunks) {
            ProcessorId processor_id = chunk.owner_id;

            auto instance = TaskInstance {
                .worker = worker,
                .graph = graph,
                .chunk = chunk,
                .memory_id = system_info.affinity_memory(processor_id),
                .buffers = {},
                .dependencies = {}};

            auto task = std::make_shared<ComputeTaskImpl<Launcher, packed_argument_t<Args>...>>(
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
                .worker = worker,  //
                .graph = graph,
                .event_id = event_id,
                .dependencies = events};

            (std::get<Is>(handlers).after_submit(result), ...);
        }

        return graph.join_events(events);
    });
}
}  // namespace detail

template<typename Launcher, typename... Args>
EventId parallel_submit(
    Worker& worker,
    const SystemInfo& system_info,
    const Domain& partition,
    Launcher launcher,
    Args&&... args
) {
    return detail::parallel_submit_impl(
        std::index_sequence_for<Args...> {},
        worker,
        system_info,
        partition,
        launcher,
        std::forward<Args>(args)...
    );
}

}  // namespace kmm