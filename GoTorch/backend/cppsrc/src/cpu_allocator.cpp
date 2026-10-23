#include "gotorch/allocator.hpp"

#include <cstdlib>
#include <new>

namespace gotorch {

/**
 * @brief Host CPU memory allocator utilizing standard heap allocation.
 */
class CPUAllocator : public Allocator {
public:
    /**
     * @brief Allocates heap memory.
     *
     * @param bytes Number of bytes to allocate.
     * @return void* Pointer to the allocated memory.
     * @throw std::bad_alloc If allocation fails.
     */
    void* allocate(size_t bytes) override {
        if (bytes == 0) {
            return nullptr;
        }
        void* ptr = std::malloc(bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    /**
     * @brief Deallocates host heap memory.
     *
     * @param ptr Pointer to memory previously allocated with allocate().
     */
    void deallocate(void* ptr) override {
        std::free(ptr);
    }

    /**
     * @brief Returns Device::CPU.
     *
     * @return Device Device::CPU.
     */
    Device device() const noexcept override {
        return Device::CPU;
    }
};

/**
 * @brief Returns the singleton host CPU memory allocator.
 *
 * @return Allocator* Pointer to the global CPU allocator instance.
 */
Allocator* get_cpu_allocator() {
    static CPUAllocator instance;
    return &instance;
}

}  // namespace gotorch
