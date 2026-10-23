#ifndef GOTORCH_DEVICE_HPP
#define GOTORCH_DEVICE_HPP

#include <string>

namespace gotorch {

/**
 * @brief Hardware execution device types supported by GoTorch.
 */
enum class Device {
    CPU = 0,
    GPU = 1,
};

/**
 * @brief Converts a Device enum value to its string representation.
 *
 * @param dev The device enum value.
 * @return std::string String name of the device ("cpu" or "gpu").
 */
inline std::string device_to_string(Device dev) {
    return dev == Device::GPU ? "gpu" : "cpu";
}

}  // namespace gotorch

#endif  // GOTORCH_DEVICE_HPP
