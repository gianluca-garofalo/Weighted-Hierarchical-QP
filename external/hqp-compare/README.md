# hqp-compare

Comparing test suite between different solvers (HQP, LexLS, DAQP).

## Usage

In CMake:

```cmake
include(FetchContent)
FetchContent_Declare(
    hqp-compare
    GIT_REPOSITORY "https://github.com/gianluca-garofalo/hqp-compare.git"
    GIT_TAG "main"
)
FetchContent_MakeAvailable(hqp-compare)
```
