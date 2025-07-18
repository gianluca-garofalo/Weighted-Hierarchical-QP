# Weighted-Hierarchical-QP

[![C++ CI](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-cpp.yml/badge.svg)](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-cpp.yml)
[![Python CI](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-python.yml/badge.svg)](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-python.yml)
[![Docs CI](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-docs.yml)

## Overview

Weighted-Hierarchical-QP is a sophisticated solver designed to tackle prioritized quadratic programming problems.  
It combines task-based hierarchies and matrix decomposition techniques to handle both equality and inequality constraints with high numerical stability.

## Features

- **Hierarchical Task Management**: Aggregate multiple tasks with increasing priority levels.
- **Flexible Constraint Handling**: Supports both equality and inequality constraints.
- **Robust Numerical Methods**: Uses Cholesky decompositions, orthogonal decompositions, and call graph analysis.
- **Python Bindings**: Python interface available for easy integration with Python projects.
- **Comprehensive Documentation**: Auto-generated API docs powered by Doxygen.

## Installation

### Prerequisites

- C++20-compliant compiler.
- [CMake](https://cmake.org) version 3.22 or greater.
- [Eigen3](http://eigen.tuxfamily.org) (≥3.4.0)
- (Optional) [Doxygen](http://www.doxygen.nl) and [Graphviz](https://graphviz.org) for documentation.

### Building the Project

1. Clone the repository.
2. Create a build directory inside the project root:
   ```bash
   mkdir build && cd build
   ```
3. Configure the project with CMake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
4. Build the project:
   ```bash
   cmake --build .
   ```
5. Run tests:
   ```bash
   ctest -C Release
   ```

### Python Bindings

The solver also provides Python bindings for easy integration with Python projects:

```bash
# Install Python dependencies
pip install numpy pybind11

# Build using CMake (recommended)
cmake --build --preset python-release

# Or build using setup.py
cd python
python setup.py build_ext --inplace

# Test the bindings
python test_pyhqp.py
```

## Usage

After building, you may integrate the solver as a library:
```cpp
#include <hqp/hqp.hpp>
// ...use the HierarchicalQP solver as documented in the API...
```

## Documentation Generation

This project leverages Doxygen for comprehensive API documentation. To generate the docs:

1. Ensure Doxygen (and Graphviz for diagrams) is installed.
2. Modify the Doxyfile if necessary.
3. Run the following command in the project root:
   ```bash
   doxygen Doxyfile
   ```
4. Open the generated `docs/index.html` in your browser to view detailed documentation.

## Contributing

Contributions are welcome! Please review the following guidelines:
- Fork the repository.
- Create a feature branch.
- Write clear commit messages and update documentation accordingly.
- Submit a pull request detailing your changes.

## License

This project is distributed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for more details.
