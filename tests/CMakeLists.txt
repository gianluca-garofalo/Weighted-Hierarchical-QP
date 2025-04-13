project(HQP_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)

add_executable(00_basic_select 00_basic_select.cpp library.cpp)
add_executable(01_metric_swap 01_metric_swap.cpp library.cpp)

set(TARGETS 
    00_basic_select
    01_metric_swap
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
