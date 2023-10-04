#include "kmm.hpp"

namespace kmm {

// Manager

Manager::Manager() {
    this->next_allocation = 0;
    this->next_task = 0;
    this->allocations = std::map<unsigned int, Buffer>();
    this->streams = std::map<unsigned int, Stream>();
    this->tasks = std::map<unsigned int, Task>();
}

bool Manager::stream_exist(unsigned int stream) {
    return this->streams.find(stream) != this->streams.end();
}

Task Manager::run() {
    unsigned int task_id = this->next_task++;

    this->tasks[task_id] = Task(task_id);

    return this->tasks[task_id];
}

// Buffer

Buffer::Buffer() {
    this->buffer = nullptr;
    this->size = 0;
    this->device = std::make_shared<UnknownDevice>();
    this->memory = std::make_shared<DefaultMemory>();
}

Buffer::Buffer(std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<UnknownDevice>();
    this->memory = std::make_shared<DefaultMemory>();
}

Buffer::Buffer(std::size_t size, CUDAPinned& memory) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<CPU>();
    this->memory = std::make_shared<CUDAPinned>();
}

Buffer::Buffer(CPU& device, std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<CPU>();
    this->memory = std::make_shared<DefaultMemory>();
}

Buffer::Buffer(CUDA& device, std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<CUDA>(device.device_id);
    this->memory = std::make_shared<DefaultMemory>();
}

std::size_t Buffer::getSize() const {
    return this->size;
}

void Buffer::setSize(std::size_t size) {
    this->size = size;
}

std::shared_ptr<DeviceType> Buffer::getDevice() {
    return this->device;
}

void Buffer::setDevice(CPU& device) {
    this->device = std::make_shared<CPU>();
}

void Buffer::setDevice(CUDA& device) {
    this->device = std::make_shared<CUDA>(device.device_id);
}

std::shared_ptr<MemoryType> Buffer::getMemory() {
    return this->memory;
}

void Buffer::setMemory(CUDAPinned& memory) {
    this->memory = std::make_shared<CUDAPinned>();
}

bool Buffer::is_allocated() const {
    return this->buffer != nullptr;
}

void Buffer::allocate() {
    this->buffer = malloc(this->size);
}

void Buffer::destroy() {
    free(this->buffer);
    this->buffer = nullptr;
    this->size = 0;
    this->device = std::make_shared<UnknownDevice>();
}

void* Buffer::getPointer() {
    return this->buffer;
}

unsigned int* Buffer::getPointer(UInteger& type) {
    return reinterpret_cast<unsigned int*>(this->buffer);
}

int* Buffer::getPointer(Integer& type) {
    return reinterpret_cast<int*>(this->buffer);
}

float* Buffer::getPointer(FP_Single& type) {
    return reinterpret_cast<float*>(this->buffer);
}

double* Buffer::getPointer(FP_Double& type) {
    return reinterpret_cast<double*>(this->buffer);
}

// GPU

GPU::GPU() {
    this->device_id = 0;
}

GPU::GPU(unsigned int device_id) {
    this->device_id = device_id;
}

// Task

Task::Task() {
    this->id = 0;
}

Task::Task(unsigned int id) {
    this->id = id;
}

}  // namespace kmm
