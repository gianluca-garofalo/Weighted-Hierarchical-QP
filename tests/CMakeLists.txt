include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(daqp
    GIT_REPOSITORY "https://github.com/gianluca-garofalo/daqp.git"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(daqp)

FetchContent_Declare(lexls
    GIT_REPOSITORY "https://github.com/jrl-umi3218/lexls"
    GIT_TAG "origin/master"
)
FetchContent_MakeAvailable(lexls)


project(HQP_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)

add_executable(00_basic_select 00_basic_select.cpp library.cpp)
add_executable(01_metric_swap 01_metric_swap.cpp library.cpp)
add_executable(02_compare_solvers 02_compare_solvers.cpp library.cpp ${daqp_SOURCE_DIR}/interfaces/daqp-eigen/daqp.cpp)
add_executable(03_set_stack 03_set_stack.cpp)
add_executable(04_random_problem 04_random_problem.cpp)

set(TARGETS 
    00_basic_select
    01_metric_swap
	03_set_stack
)

foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} Eigen3::Eigen HQP::HQP)
	target_include_directories(
	    ${TARGET}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}
	)
	
	add_test(NAME test_${TARGET} COMMAND ${TARGET})
endforeach()

set(TARGETS 
    02_compare_solvers
    04_random_problem
)

foreach(TARGET ${TARGETS})
	target_link_libraries(${TARGET} Eigen3::Eigen HQP::HQP daqp_eigen lexls)
	target_include_directories(
		${TARGET}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}
	)

	add_test(NAME test_${TARGET} COMMAND ${TARGET})
endforeach()
