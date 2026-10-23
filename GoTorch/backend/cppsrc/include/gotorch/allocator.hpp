#ifndef GOTORCH_ALLOCATOR_HPP
#define GOTORCH_ALLOCATOR_HPP

#include <cstddef>

#include "gotorch/device.hpp"

namespace gotorch {

/**
 * @brief Abstract base strategy interface for memory allocation and deallocation.
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    /**
     * @brief Allocates a contiguous block of bytes on the target device.
     *
     * @param bytes Total number of bytes to allocate.
     * @return void* Pointer to the allocated memory buffer.
     * @throw std::bad_alloc If memory allocation fails.
     */
    virtual void* allocate(size_t bytes) = 0;

    /**
     * @brief Deallocates a previously allocated memory buffer.
     *
     * @param ptr Pointer to the allocated memory buffer.
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Returns the device associated with this allocator.
     *
     * @return Device Device enum identifying the memory residence.
     */
    virtual Device device() const noexcept = 0;
};

/**
 * @brief Returns the singleton host CPU memory allocator.
 *
 * @return Allocator* Pointer to the global CPU allocator instance.
 */
Allocator* get_cpu_allocator();

/**
 * @brief Returns the singleton AMD ROCm/HIP GPU memory allocator.
 *
 * @return Allocator* Pointer to the global GPU allocator instance.
 */
Allocator* get_gpu_allocator();

}  // namespace gotorch

#endif  // GOTORCH_ALLOCATOR_HPP
