include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

project(HQP_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)

add_executable(07_template_demo 07_template_demo.cpp)
add_executable(08_warm_start 08_warm_start.cpp)

set(TARGETS
    07_template_demo
    08_warm_start
)

foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen HQP::HQP)
    target_include_directories(
        ${TARGET}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_test(NAME test_${TARGET} COMMAND ${TARGET})
endforeach()

# Import external hqp-tasks feature
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-tasks/CMakeLists.txt")
    get_filename_component(HQP_TASKS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-tasks" ABSOLUTE)
    set(FETCHCONTENT_SOURCE_DIR_HQP-TASKS "${HQP_TASKS_DIR}" CACHE PATH "Path to hqp-tasks source" FORCE)
    set(FETCHCONTENT_SOURCE_DIR_HQP_TASKS "${HQP_TASKS_DIR}" CACHE PATH "Path to hqp-tasks source" FORCE)
endif()

FetchContent_Declare(hqp-tasks
    GIT_REPOSITORY "https://github.com/gianluca-garofalo/hqp-tasks.git"
    GIT_TAG "main"
)
FetchContent_MakeAvailable(hqp-tasks)

# Import external hqp-compare feature
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-compare/CMakeLists.txt")
    get_filename_component(HQP_COMPARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-compare" ABSOLUTE)
    set(FETCHCONTENT_SOURCE_DIR_HQP-COMPARE "${HQP_COMPARE_DIR}" CACHE PATH "Path to hqp-compare source" FORCE)
    set(FETCHCONTENT_SOURCE_DIR_HQP_COMPARE "${HQP_COMPARE_DIR}" CACHE PATH "Path to hqp-compare source" FORCE)
endif()

FetchContent_Declare(hqp-compare
    GIT_REPOSITORY "https://github.com/gianluca-garofalo/hqp-compare.git"
    GIT_TAG "main"
)
FetchContent_MakeAvailable(hqp-compare)
