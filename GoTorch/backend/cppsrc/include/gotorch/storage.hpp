#ifndef GOTORCH_STORAGE_HPP
#define GOTORCH_STORAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "gotorch/allocator.hpp"
#include "gotorch/device.hpp"
#include "gotorch/device_transfer.hpp"

namespace gotorch {

/**
 * @brief Concrete non-virtual physical memory container decoupled from allocation strategy.
 *
 * Encapsulates a raw typed pointer, element count, target device, and an Allocator strategy pointer.
 * Provides inlined direct pointer access with zero virtual dispatch overhead.
 */
template <typename T = float>
class Storage {
private:
    Allocator* allocator_{nullptr};
    Device device_{Device::CPU};
    size_t total_elements_{0};
    T* data_ptr_{nullptr};

public:
    /**
     * @brief Default constructor creating an empty host storage.
     */
    Storage() : Storage(0, nullptr) {}

    /**
     * @brief Allocates an uninitialized buffer on the target device.
     *
     * @param size Number of elements to allocate.
     * @param alloc Strategy allocator (defaults to CPU allocator if nullptr).
     */
    explicit Storage(size_t size, Allocator* alloc = nullptr)
        : allocator_(alloc ? alloc : get_cpu_allocator()),
          device_(allocator_->device()),
          total_elements_(size),
          data_ptr_(static_cast<T*>(allocator_->allocate(size * sizeof(T)))) {}

    /**
     * @brief Allocates buffer initialized with a specific value.
     *
     * @param size Number of elements to allocate.
     * @param init_val Initial value for all elements.
     * @param alloc Strategy allocator (defaults to CPU allocator if nullptr).
     */
    Storage(size_t size, const T& init_val, Allocator* alloc = nullptr)
        : Storage(size, alloc) {
        if (data_ptr_) {
            if (device_ == Device::CPU) {
                std::fill(data_ptr_, data_ptr_ + total_elements_, init_val);
            } else {
                std::vector<T> host_fill(total_elements_, init_val);
                copy_memory(
                    data_ptr_,
                    device_,
                    host_fill.data(),
                    Device::CPU,
                    total_elements_ * sizeof(T)
                );
            }
        }
    }

    /**
     * @brief Constructs storage by copying from a contiguous pointer.
     *
     * @param ptr Source pointer (must be valid on host if alloc is CPU).
     * @param size Number of elements to copy.
     * @param alloc Strategy allocator (defaults to CPU allocator if nullptr).
     */
    Storage(const T* ptr, size_t size, Allocator* alloc = nullptr)
        : Storage(size, alloc) {
        if (data_ptr_ && ptr) {
            copy_memory(
                data_ptr_,
                device_,
                ptr,
                Device::CPU,
                total_elements_ * sizeof(T)
            );
        }
    }

    /**
     * @brief Constructs storage by copying from an std::vector.
     *
     * @param vec Source vector on the host.
     * @param alloc Strategy allocator (defaults to CPU allocator if nullptr).
     */
    explicit Storage(const std::vector<T>& vec, Allocator* alloc = nullptr)
        : Storage(vec.data(), vec.size(), alloc) {}

    /**
     * @brief Destructor deallocating buffer via strategy allocator.
     */
    ~Storage() {
        if (data_ptr_ && allocator_) {
            allocator_->deallocate(data_ptr_);
            data_ptr_ = nullptr;
        }
    }

    /**
     * @brief Deep copy constructor allocating new buffer on the same device.
     *
     * @param other Storage to copy from.
     */
    Storage(const Storage& other)
        : allocator_(other.allocator_),
          device_(other.device_),
          total_elements_(other.total_elements_),
          data_ptr_(nullptr) {
        if (total_elements_ > 0 && other.data_ptr_) {
            data_ptr_ = static_cast<T*>(allocator_->allocate(total_elements_ * sizeof(T)));
            copy_memory(
                data_ptr_,
                device_,
                other.data_ptr_,
                other.device_,
                total_elements_ * sizeof(T)
            );
        }
    }

    /**
     * @brief Deep copy assignment operator.
     *
     * @param other Storage to copy from.
     * @return Storage& Reference to this storage.
     */
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            if (data_ptr_ && allocator_) {
                allocator_->deallocate(data_ptr_);
                data_ptr_ = nullptr;
            }
            total_elements_ = other.total_elements_;
            device_ = other.device_;
            allocator_ = other.allocator_;
            if (total_elements_ > 0 && other.data_ptr_) {
                data_ptr_ = static_cast<T*>(allocator_->allocate(total_elements_ * sizeof(T)));
                copy_memory(
                    data_ptr_,
                    device_,
                    other.data_ptr_,
                    other.device_,
                    total_elements_ * sizeof(T)
                );
            }
        }
        return *this;
    }

    /**
     * @brief Move constructor transferring pointer ownership.
     *
     * @param other Storage to move from.
     */
    Storage(Storage&& other) noexcept
        : allocator_(other.allocator_),
          device_(other.device_),
          total_elements_(other.total_elements_),
          data_ptr_(other.data_ptr_) {
        other.data_ptr_ = nullptr;
        other.total_elements_ = 0;
    }

    /**
     * @brief Move assignment operator.
     *
     * @param other Storage to move from.
     * @return Storage& Reference to this storage.
     */
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (data_ptr_ && allocator_) {
                allocator_->deallocate(data_ptr_);
            }
            data_ptr_ = other.data_ptr_;
            total_elements_ = other.total_elements_;
            device_ = other.device_;
            allocator_ = other.allocator_;
            other.data_ptr_ = nullptr;
            other.total_elements_ = 0;
        }
        return *this;
    }

    /**
     * @brief Direct inlined pointer access to the buffer.
     *
     * @return T* Raw typed pointer to the start of the buffer.
     */
    T* data() noexcept {
        return data_ptr_;
    }

    /**
     * @brief Const direct inlined pointer access to the buffer.
     *
     * @return const T* Const raw typed pointer to the start of the buffer.
     */
    const T* data() const noexcept {
        return data_ptr_;
    }

    /**
     * @brief Total number of elements allocated in this buffer.
     *
     * @return size_t Element count.
     */
    size_t size() const noexcept {
        return total_elements_;
    }

    /**
     * @brief Device residency of this storage buffer.
     *
     * @return Device Target device.
     */
    Device device() const noexcept {
        return device_;
    }

    /**
     * @brief Strategy allocator associated with this storage buffer.
     *
     * @return Allocator* Pointer to the allocator strategy.
     */
    Allocator* allocator() const noexcept {
        return allocator_;
    }
};

}  // namespace gotorch

#endif  // GOTORCH_STORAGE_HPP
