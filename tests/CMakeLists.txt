find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)

# Solver-only tests. The stack-of-tasks tests live in hqp-tasks and the
# comparisons against other solvers (DAQP, LexLS) live in hqp-compare.
set(TARGETS
    03_set_stack
    07_template_demo
    08_warm_start
    09_equality_lock
    10_ratio_test
)

foreach(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen HQP::HQP)
    add_test(NAME test_${TARGET} COMMAND ${TARGET})
endforeach()
