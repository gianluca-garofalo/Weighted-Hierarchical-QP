cmake_minimum_required(VERSION 3.22)

project(HQP VERSION 0.1.1 DESCRIPTION "A HQP solver" LANGUAGES CXX)
message(PROJECT_NAME="${PROJECT_NAME}")
message(PROJECT_VERSION="${PROJECT_VERSION}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
message(CMAKE_INSTALL_INCLUDEDIR="${CMAKE_INSTALL_INCLUDEDIR}")
message(CMAKE_CURRENT_LIST_DIR="${CMAKE_CURRENT_LIST_DIR}")
message(CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_DEBUG_POSTFIX deb)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


file(GLOB SRC_FILES src/*.cpp)
message(SRC_FILES="${SRC_FILES}")
add_library(${PROJECT_NAME} ${SRC_FILES})
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS ${PROJECT_NAME})

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_compile_features(${PROJECT_NAME}	PUBLIC cxx_std_20)


if(BUILD_TESTING)
	add_subdirectory(tests)
endif()


install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(doc)

write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}_config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
)

install(
    EXPORT ${PROJECT_NAME}_targets
    FILE ${PROJECT_NAME}_targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
)
