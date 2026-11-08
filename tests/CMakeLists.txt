include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

project(HQP_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)

add_executable(03_set_stack 03_set_stack.cpp)
add_executable(07_template_demo 07_template_demo.cpp)
add_executable(08_warm_start 08_warm_start.cpp)

set(TARGETS
    03_set_stack
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

# Option to fetch external test repositories
option(HQP_FETCH_EXTERNAL_TESTS "Fetch external test repositories (hqp-tasks, hqp-compare)" ON)

if(HQP_FETCH_EXTERNAL_TESTS)
    # Check availability of hqp-tasks
    set(HQP_TASKS_AVAILABLE FALSE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../external/hqp-tasks/CMakeLists.txt")
        get_filename_component(HQP_TASKS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/hqp-tasks" ABSOLUTE)
        set(FETCHCONTENT_SOURCE_DIR_HQP-TASKS "${HQP_TASKS_DIR}" CACHE PATH "Path to hqp-tasks source" FORCE)
        set(FETCHCONTENT_SOURCE_DIR_HQP_TASKS "${HQP_TASKS_DIR}" CACHE PATH "Path to hqp-tasks source" FORCE)
        set(HQP_TASKS_AVAILABLE TRUE)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-tasks/CMakeLists.txt")
        get_filename_component(HQP_TASKS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-tasks" ABSOLUTE)
        set(FETCHCONTENT_SOURCE_DIR_HQP-TASKS "${HQP_TASKS_DIR}" CACHE PATH "Path to hqp-tasks source" FORCE)
        set(FETCHCONTENT_SOURCE_DIR_HQP_TASKS "${HQP_TASKS_DIR}" CACHE PATH "Path to hqp-tasks source" FORCE)
        set(HQP_TASKS_AVAILABLE TRUE)
    elseif(DEFINED FETCHCONTENT_SOURCE_DIR_HQP-TASKS OR DEFINED FETCHCONTENT_SOURCE_DIR_HQP_TASKS)
        set(HQP_TASKS_AVAILABLE TRUE)
    else()
        execute_process(
            COMMAND git ls-remote "https://github.com/gianluca-garofalo/hqp-tasks.git" refs/heads/main
            RESULT_VARIABLE HQP_TASKS_REMOTE_RES
            OUTPUT_VARIABLE HQP_TASKS_REMOTE_OUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            TIMEOUT 5
        )
        if(HQP_TASKS_REMOTE_RES EQUAL 0 AND NOT "${HQP_TASKS_REMOTE_OUT}" STREQUAL "")
            set(HQP_TASKS_AVAILABLE TRUE)
        endif()
    endif()

    if(HQP_TASKS_AVAILABLE)
        FetchContent_Declare(hqp-tasks
            GIT_REPOSITORY "https://github.com/gianluca-garofalo/hqp-tasks.git"
            GIT_TAG "main"
        )
        FetchContent_MakeAvailable(hqp-tasks)
    else()
        message(STATUS "hqp-tasks not available (local source not found and remote main branch not published). Skipping hqp-tasks tests.")
    endif()

    # Check availability of hqp-compare
    set(HQP_COMPARE_AVAILABLE FALSE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../external/hqp-compare/CMakeLists.txt")
        get_filename_component(HQP_COMPARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/hqp-compare" ABSOLUTE)
        set(FETCHCONTENT_SOURCE_DIR_HQP-COMPARE "${HQP_COMPARE_DIR}" CACHE PATH "Path to hqp-compare source" FORCE)
        set(FETCHCONTENT_SOURCE_DIR_HQP_COMPARE "${HQP_COMPARE_DIR}" CACHE PATH "Path to hqp-compare source" FORCE)
        set(HQP_COMPARE_AVAILABLE TRUE)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-compare/CMakeLists.txt")
        get_filename_component(HQP_COMPARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hqp-compare" ABSOLUTE)
        set(FETCHCONTENT_SOURCE_DIR_HQP-COMPARE "${HQP_COMPARE_DIR}" CACHE PATH "Path to hqp-compare source" FORCE)
        set(FETCHCONTENT_SOURCE_DIR_HQP_COMPARE "${HQP_COMPARE_DIR}" CACHE PATH "Path to hqp-compare source" FORCE)
        set(HQP_COMPARE_AVAILABLE TRUE)
    elseif(DEFINED FETCHCONTENT_SOURCE_DIR_HQP-COMPARE OR DEFINED FETCHCONTENT_SOURCE_DIR_HQP_COMPARE)
        set(HQP_COMPARE_AVAILABLE TRUE)
    else()
        execute_process(
            COMMAND git ls-remote "https://github.com/gianluca-garofalo/hqp-compare.git" refs/heads/main
            RESULT_VARIABLE HQP_COMPARE_REMOTE_RES
            OUTPUT_VARIABLE HQP_COMPARE_REMOTE_OUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            TIMEOUT 5
        )
        if(HQP_COMPARE_REMOTE_RES EQUAL 0 AND NOT "${HQP_COMPARE_REMOTE_OUT}" STREQUAL "")
            set(HQP_COMPARE_AVAILABLE TRUE)
        endif()
    endif()

    if(HQP_COMPARE_AVAILABLE)
        FetchContent_Declare(hqp-compare
            GIT_REPOSITORY "https://github.com/gianluca-garofalo/hqp-compare.git"
            GIT_TAG "main"
        )
        FetchContent_MakeAvailable(hqp-compare)
    else()
        message(STATUS "hqp-compare not available (local source not found and remote main branch not published). Skipping hqp-compare tests.")
    endif()
endif()
