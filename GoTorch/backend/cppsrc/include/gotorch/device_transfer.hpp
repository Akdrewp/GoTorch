#ifndef GOTORCH_DEVICE_TRANSFER_HPP
#define GOTORCH_DEVICE_TRANSFER_HPP

#include <cstddef>

#include "gotorch/device.hpp"

namespace gotorch {

/**
 * @brief Copies a contiguous memory block between devices.
 *
 * 1. If both devices are CPU, perform a direct standard memory copy.
 * 2. If at least one device is GPU, route to the HIP runtime memcpy bridge.
 *
 * @param dst Destination memory pointer.
 * @param dst_dev Target destination device.
 * @param src Source memory pointer.
 * @param src_dev Source origin device.
 * @param bytes Number of bytes to transfer.
 * @throw std::runtime_error If the device transfer fails.
 */
void copy_memory(void* dst, Device dst_dev, const void* src, Device src_dev, size_t bytes);

}  // namespace gotorch

#endif  // GOTORCH_DEVICE_TRANSFER_HPP
