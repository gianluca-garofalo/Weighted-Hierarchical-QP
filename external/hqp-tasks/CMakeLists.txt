cmake_minimum_required(VERSION 3.22)
project(hqp-tasks VERSION 0.1.0 DESCRIPTION "Stack of Tasks library for HQP" LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)
option(BUILD_TESTING "Build testing" ON)

include(CTest)
include(GNUInstallDirs)

add_library(hqp-tasks INTERFACE)
add_library(hqp-tasks::hqp-tasks ALIAS hqp-tasks)

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(hqp-tasks INTERFACE Eigen3::Eigen)

target_include_directories(hqp-tasks
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(hqp-tasks INTERFACE cxx_std_20)

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

install(
    TARGETS hqp-tasks
    EXPORT hqp-tasks_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.tpp"
)
