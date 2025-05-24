/**
 * @file utils.hpp
 * @brief Declares common utility functions used throughout the HQP solver.
 *
 * The functions declared in this file assist with array and index manipulation,
 * which are critical for processing constraints and handling active sets.
 */
#ifndef _UtilsHQP_
#define _UtilsHQP_

#include <memory>
#include <type_traits>
#include <Eigen/Dense>

namespace hqp {

/**
 * @brief Identifies and returns the indices of all true elements in a given boolean array.
 *
 * Iterates over the array and collects indices where the condition evaluates to true.
 *
 * @param in Boolean array to be scanned.
 * @return VectorXi containing the positions of true values.
 */
Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>&);

// GenericPtr wrapper around std::shared_ptr or std::unique_ptr
template<template<typename> class SmartPtr, typename T>
class GenericPtr : public SmartPtr<T> {
  private:
    // Helper function to create the correct smart pointer type
    template<typename Derived, typename... Args>
    static SmartPtr<T> make_ptr(Args&&... args) {
        if constexpr (std::is_same_v<SmartPtr<T>, std::shared_ptr<T>>) {
            return std::make_shared<Derived>(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<SmartPtr<T>, std::unique_ptr<T>>) {
            return std::make_unique<Derived>(std::forward<Args>(args)...);
        } else {
            static_assert(sizeof(SmartPtr<T>) == 0, "Unsupported smart pointer type");
        }
    }

  public:
    using SmartPtr<T>::SmartPtr;  // Inherit constructors

    template<typename Derived = T,
             typename... Args,
             // Only enable if there are arguments AND (either T is not abstract or Derived is not T)
             typename = std::enable_if_t<(sizeof...(Args) > 0) && (!std::is_abstract_v<T> || !std::is_same_v<Derived, T>)>>
    GenericPtr(Args&&... args)
      : SmartPtr<T>(make_ptr<Derived>(std::forward<Args>(args)...)) {
    }

    template <typename Derived>
    Derived* cast() {
        return static_cast<Derived*>(SmartPtr<T>::get());
    }
};

// SmartContainer of GenericPtr
template<template<typename, typename> class Container,
         template<template<typename> class, typename> class PtrWrapper,  // Wrapper class (GenericPtr)
         template<typename> class SmartPtr,  // Smart pointer type (std::shared_ptr or std::unique_ptr)
         typename T>
class SmartContainer : public Container<PtrWrapper<SmartPtr, T>, std::allocator<PtrWrapper<SmartPtr, T>>> {
  public:
    using Base = Container<PtrWrapper<SmartPtr, T>, std::allocator<PtrWrapper<SmartPtr, T>>>;
    using Base::Base;  // Inherit constructors

    // emplace_back automatically creates the smart pointer inside GenericPtr
    template<typename Derived, typename... Args>
    void emplace_back(Args&&... args) {
        static_assert(std::is_base_of_v<T, Derived>, "Derived must be a subclass of T");
        Base::emplace_back(PtrWrapper<SmartPtr, Derived>(std::forward<Args>(args)...));
    }
};

}  // namespace hqp

#endif  // _UtilsHQP_
