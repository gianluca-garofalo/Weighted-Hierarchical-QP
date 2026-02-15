cmake_minimum_required(VERSION 3.22)

project(pyhqp VERSION 0.1.1 DESCRIPTION "Python bindings for HQP solver" LANGUAGES CXX)

# Check if we're being built as part of the main project or standalone
if(NOT TARGET HQP::HQP)
    # We're being built standalone, so we need to find the HQP library
    message(STATUS "Building Python bindings standalone")
    
    # Add the parent directory include path
    get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
    set(HQP_INCLUDE_DIR "${PARENT_DIR}/include")
    
    if(NOT EXISTS "${HQP_INCLUDE_DIR}/hqp/hqp.hpp")
        message(FATAL_ERROR "HQP headers not found. Expected at: ${HQP_INCLUDE_DIR}/hqp/hqp.hpp")
    endif()
    
    # Create interface target for HQP
    add_library(HQP INTERFACE)
    add_library(HQP::HQP ALIAS HQP)
    
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(HQP INTERFACE Eigen3::Eigen)
    
    target_include_directories(HQP INTERFACE ${HQP_INCLUDE_DIR})
    target_compile_features(HQP INTERFACE cxx_std_20)
    
    set(HQP_TARGET HQP::HQP)
else()
    # We're being built as part of the main project
    message(STATUS "Building Python bindings as part of main project")
    set(HQP_TARGET HQP)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find pybind11 using different methods
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Try to get pybind11 from Python installation
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_PYTHON_RESULT
    )
    if(PYBIND11_PYTHON_RESULT EQUAL 0)
        set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
        message(STATUS "Found pybind11 via Python: ${PYBIND11_CMAKE_DIR}")
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# Create the Python extension module
pybind11_add_module(pyhqp pybind_hqp.cpp)

target_link_libraries(pyhqp PRIVATE ${HQP_TARGET})

target_compile_features(pyhqp PRIVATE cxx_std_20)

set_target_properties(pyhqp PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Add compiler flags for better debugging/optimization
target_compile_options(pyhqp PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-g -O0>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -DNDEBUG>
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Zi /Od>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2 /DNDEBUG>
)

# Install the Python module
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    install(TARGETS pyhqp DESTINATION ${PYTHON_SITE_PACKAGES})
else()
    install(TARGETS pyhqp DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

# Optionally enable testing
if(BUILD_TESTING OR CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    enable_testing()
    
    # Add test
    add_test(
        NAME test_pyhqp
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_pyhqp.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Set the Python path for the test
    set_tests_properties(test_pyhqp PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
