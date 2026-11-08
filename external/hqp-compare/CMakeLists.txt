cmake_minimum_required(VERSION 3.22)
project(hqp-compare VERSION 0.1.0 DESCRIPTION "Solver Comparison Suite for HQP" LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)
option(BUILD_TESTING "Build comparison benchmarks" ON)

include(CTest)
include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(daqp
    GIT_REPOSITORY "https://github.com/darnstrom/daqp"
    GIT_TAG "master"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(daqp)

FetchContent_Declare(lexls
    GIT_REPOSITORY "https://github.com/jrl-umi3218/lexls"
    GIT_TAG "master"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(lexls)

find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
