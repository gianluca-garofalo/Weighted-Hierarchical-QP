project(hqp_compare_tests)

find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)

add_executable(02_compare_solvers 02_compare_solvers.cpp)
add_executable(04_random_problem 04_random_problem.cpp)
add_executable(05_lex_slacks 05_lex_slacks.cpp)
add_executable(06_timing_stats 06_timing_stats.cpp)

set(TARGETS
    02_compare_solvers
    04_random_problem
    05_lex_slacks
    06_timing_stats
)

foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen daqp_eigen lexls)
    if(TARGET HQP::HQP)
        target_link_libraries(${TARGET} PRIVATE HQP::HQP)
    endif()
    if(TARGET hqp-tasks)
        target_link_libraries(${TARGET} PRIVATE hqp-tasks)
    endif()
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME test_${TARGET} COMMAND ${TARGET})
endforeach()
