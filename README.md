# HQP: Hierarchical Quadratic Programming

[![C++ CI](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-cpp.yml/badge.svg)](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-cpp.yml)
[![Python CI](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-python.yml/badge.svg)](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-python.yml)
[![Docs CI](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP/actions/workflows/ci-docs.yml)


## What is HQP?

HQP (Hierarchical Quadratic Programming) is a solver for problems where you have several objectives or constraints, and some are more important than others. You tell it what matters most, and it finds the best solution that respects your priorities.

If you work in robotics, control, or any field where you need to balance multiple goals, you know this is a problem of logic and mathematics—not magic. HQP is designed to make that process clear, reliable, and efficient.

## Why Use HQP?


### Hierarchical Task Management
- Prioritize tasks naturally: HQP lets you specify which constraints matter most, and solves them in order.
- Handles both equality and inequality constraints, so you can model real-world problems accurately.

### Robust Numerical Methods
- Uses proven techniques (Cholesky, orthogonal decompositions) for stability and speed.
- Built-in anti-cycling and adaptive tolerances to keep the solver reliable, even for tough problems.

### Transparent Status Reporting
- Clear feedback on why the solver stopped: success, timeout, cycling, or other conditions.
- Performance metrics and debug output designed to help you understand and improve your models.

### Performance Optimizations
- Iteration limits and timeouts that adapt to your problem size.
- Detects stagnation and cycling, so you don’t waste time on unsolvable or ill-posed problems.



## Quick Start

### Prerequisites

- **C++20** compiler
- **CMake** ≥3.22
- **Eigen3** ≥3.4.0
- **Python** (optional, for Python bindings)
- **Doxygen** + **Graphviz** (optional, for docs)

### Building

```bash
git clone https://github.com/your-repo/HQP.git
cd HQP
mkdir build && cd build
cmake --preset=release
cmake --build .
ctest -C Release
```

### Python Bindings

```bash
cmake --preset=python-release
cmake --build .
python -c "import hqp; print('Python bindings work!')"
```

breaks = np.array([3, 7, 10])
breaks = np.array([3, 7, 10])

## Usage Examples

### C++ Example

```cpp
#include <hqp/hqp.hpp>
#include <Eigen/Dense>

int main() {
    hqp::HierarchicalQP solver(10, 5);  // 10 constraints, 5 variables
    Eigen::MatrixXd A(10, 5);
    Eigen::VectorXd bl(10), bu(10);
    Eigen::VectorXi breaks(3);
    // ... fill your matrices with actual data ...
    solver.set_problem(A, bl, bu, breaks);
    auto solution = solver.get_primal();
    if (solver.is_solved()) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << solver.get_status_message() << std::endl;
    }
    return 0;
}
```

### Python Example

```python
import hqp
import numpy as np

solver = hqp.HierarchicalQP(10, 5)
A = np.random.rand(10, 5)
bl = -np.ones(10)
bu = np.ones(10)
breaks = np.array([3, 7, 10])
solver.set_problem(A, bl, bu, breaks)
solution = solver.get_primal()
print(f"Status: {solver.get_status_message()}")
print(f"Solved successfully: {solver.is_solved()}")
```



## Configuration Options

You can adjust solver behavior with these CMake options:

```bash
cmake --preset=release -DHQP_TIMEOUT_SECONDS=10.0         # Set a faster timeout
cmake --preset=release -DHQP_MAX_ITERATIONS=5000          # Allow more iterations
cmake --preset=release -DHQP_ANTI_CYCLING_BUFFER_SIZE=100 # Use a bigger anti-cycling buffer
cmake --preset=debug -DHQP_STAGNATION_THRESHOLD=5         # Lower stagnation threshold for debugging
```



## Status Codes

| Status | What It Means |
|--------|---------------|
| `SUCCESS` | Solution found |
| `MAX_ITERATIONS_REACHED` | Maximum iterations reached |
| `TIMEOUT_REACHED` | Timeout reached |
| `CYCLING_DETECTED` | Solver detected cycling |
| `STAGNATION_DETECTED` | Solver detected stagnation |
| `NUMERICAL_ISSUE` | Numerical issue encountered |
| `INFEASIBLE` | No solution exists |
| `UNBOUNDED` | Solution is unbounded |




## Testing

```bash
ctest --preset=release                       # Run all tests
./build/tests/HQP_tests                      # Run a specific test (after preset build)
ctest --preset=release --output-on-failure   # Run with verbose output
```




## Documentation

### Generate Docs

```bash
cmake --build --preset=release --target doc
open build/doc/html/index.html
```

### API Reference

- **Main Class**: `hqp::HierarchicalQP` - Main solver class
- **Status Enum**: `hqp::SolverStatus` - Solver status codes
- **Info Struct**: `hqp::SolverInfo` - Solver details and stats



## Contributing

Contributions are welcome. If you find a bug or want to add a feature:

1. Fork the repo
2. Create a feature branch
3. Write tests
4. Open a pull request


### Development Setup

```bash
git clone https://github.com/your-username/HQP.git
cd HQP
cmake --preset=debug
cmake --build --preset=debug
ctest --preset=debug
```



## Troubleshooting

### Common Issues

**CMake can't find Eigen**
```bash
sudo apt-get install libeigen3-dev  # Ubuntu/Debian
brew install eigen                  # macOS
```

**Solver times out**
```bash
cmake --preset=release -DHQP_TIMEOUT_SECONDS=60.0
```

**Python bindings don't work**
```bash
python --version
pip install numpy pybind11
```



## License

BSD 3-Clause License.



## What's New

- Performance improvements
- More detailed status reporting
- Adaptive tolerances
- Anti-cycling mechanisms
- Timeout protection
- Debug output that actually helps
- Easier CMake configuration
- Better error messages

---

*"In a world full of QP solvers, be the one that actually works."*
