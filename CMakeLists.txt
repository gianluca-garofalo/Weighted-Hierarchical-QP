cmake_minimum_required(VERSION 3.4)

# set(CMAKE_INSTALL_PREFIX "/home/deshi") cat install_manifest.txt | xargs echo sudo rm | sh
project(HQP)

# set(CMAKE_BUILD_TYPE Release CACHE STRING "Release RelWithDebInfo Debug DEFAULT: Release" FORCE)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")

option(COMPILE_TESTS "compile the tests" ON)

add_compile_options(
  -O3 -g -Wall -Wpedantic -fPIC -std=c++20 -Wno-terminate
  )

################################################################################

file(GLOB SRC_Hqp hqp/*.cpp)
file(GLOB INC_Hqp hqp/*.hpp)

add_library(hqp SHARED
    ${SRC_Hqp}
  )

################################################################################

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)
target_link_libraries(hqp Eigen3::Eigen)

################################################################################

set_target_properties(hqp PROPERTIES SOVERSION 0)

install(
	TARGETS hqp
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
install(FILES ${INC_Hqp} DESTINATION "include/hqp")

if(${COMPILE_TESTS})
         add_subdirectory(tests)
endif()
