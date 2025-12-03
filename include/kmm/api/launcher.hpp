#pragma once

#include "kmm/core/domain.hpp"
#include "kmm/core/resource.hpp"

namespace kmm {

template<typename F>
struct Host {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    Host(F fun) : m_fun(fun) {}

    template<typename... Args>
    void operator()(Resource& resource, DomainChunk chunk, Args... args) {
        m_fun(args...);
    }

  private:
    std::decay_t<F> m_fun;
};

template<typename F>
struct GPU {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Device;

    GPU(F fun) : m_fun(fun) {}

    template<typename... Args>
    void operator()(Resource& resource, DomainChunk chunk, Args... args) {
        m_fun(resource.cast<DeviceResource>(), args...);
    }

  private:
    std::decay_t<F> m_fun;
};

template<typename F>
struct GPUKernel {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Device;

    GPUKernel(F kernel, dim3 block_size) : GPUKernel(kernel, block_size, block_size) {}

    GPUKernel(F kernel, dim3 block_size, dim3 elements_per_block, uint32_t shared_memory = 0) :
        kernel(kernel),
        block_size(block_size),
        elements_per_block(elements_per_block),
        shared_memory(shared_memory) {}

    template<typename... Args>
    void operator()(Resource& resource, DomainChunk chunk, Args... args) {
        int64_t g[3] = {
            chunk.size.get_or_default(0),
            chunk.size.get_or_default(1),
            chunk.size.get_or_default(2)
        };
        int64_t b[3] = {elements_per_block.x, elements_per_block.y, elements_per_block.z};
        dim3 grid_dim = {
            checked_cast<unsigned int>((g[0] / b[0]) + int64_t(g[0] % b[0] != 0)),
            checked_cast<unsigned int>((g[1] / b[1]) + int64_t(g[1] % b[1] != 0)),
            checked_cast<unsigned int>((g[2] / b[2]) + int64_t(g[2] % b[2] != 0)),
        };

        resource.cast<DeviceResource>().launch(  //
            grid_dim,
            block_size,
            shared_memory,
            kernel,
            args...
        );
    }

  private:
    std::decay_t<F> kernel;
    dim3 block_size;
    dim3 elements_per_block;
    uint32_t shared_memory;
};

}  // namespace kmm
