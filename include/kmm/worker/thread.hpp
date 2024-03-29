#pragma once

#include <memory>
#include <thread>

namespace kmm {

class Worker;

class WorkerThread {
  public:
    WorkerThread(std::shared_ptr<Worker> worker);
    ~WorkerThread();
    void join();

  private:
    std::thread m_thread;
};

}  // namespace kmm