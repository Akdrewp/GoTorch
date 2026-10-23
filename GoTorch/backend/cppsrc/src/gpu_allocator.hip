#include <hip/hip_runtime.h>

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "gotorch/allocator.hpp"
#include "gotorch/device_transfer.hpp"

namespace gotorch {

/**
 * @brief AMD ROCm/HIP device memory allocator utilizing hipMalloc and hipFree.
 */
class GPUAllocator : public Allocator {
public:
    /**
     * @brief Allocates device VRAM on the current HIP device.
     *
     * @param bytes Number of bytes to allocate.
     * @return void* Pointer to the allocated device memory.
     * @throw std::bad_alloc If VRAM allocation fails.
     */
    void* allocate(size_t bytes) override {
        if (bytes == 0) {
            return nullptr;
        }
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess || !ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    /**
     * @brief Deallocates device VRAM.
     *
     * @param ptr Pointer to device memory previously allocated with allocate().
     */
    void deallocate(void* ptr) override {
        if (ptr) {
            hipFree(ptr);
        }
    }

    /**
     * @brief Returns Device::GPU.
     *
     * @return Device Device::GPU.
     */
    Device device() const noexcept override {
        return Device::GPU;
    }
};

/**
 * @brief Returns the singleton AMD ROCm/HIP GPU memory allocator.
 *
 * @return Allocator* Pointer to the global GPU allocator instance.
 */
Allocator* get_gpu_allocator() {
    static GPUAllocator instance;
    return &instance;
}

/**
 * @brief Copies a contiguous memory block between devices.
 *
 * @param dst Destination memory pointer.
 * @param dst_dev Target destination device.
 * @param src Source memory pointer.
 * @param src_dev Source origin device.
 * @param bytes Number of bytes to transfer.
 * @throw std::runtime_error If the device transfer fails.
 */
void copy_memory(void* dst, Device dst_dev, const void* src, Device src_dev, size_t bytes) {
    if (bytes == 0 || dst == nullptr || src == nullptr) {
        return;
    }

    if (dst_dev == Device::CPU && src_dev == Device::CPU) {
        std::memcpy(dst, src, bytes);
        return;
    }

    hipMemcpyKind kind;
    if (src_dev == Device::CPU && dst_dev == Device::GPU) {
        kind = hipMemcpyHostToDevice;
    } else if (src_dev == Device::GPU && dst_dev == Device::CPU) {
        kind = hipMemcpyDeviceToHost;
    } else {
        kind = hipMemcpyDeviceToDevice;
    }

    hipError_t err = hipMemcpy(dst, src, bytes, kind);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("hipMemcpy failed: ") + hipGetErrorString(err)
        );
    }
}

}  // namespace gotorch
