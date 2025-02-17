project(HQP_tests)
message(PROJECT_NAME="${PROJECT_NAME}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB SRC_FILES *.cpp)
message(SRC_FILES="${SRC_FILES}")
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE HQP::HQP)
target_include_directories(
    ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
