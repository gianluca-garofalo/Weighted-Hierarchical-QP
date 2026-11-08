# hqp-tasks

Stack of Tasks library for Hierarchical Quadratic Programming (HQP).

## Overview

`hqp-tasks` provides task definitions and stack of tasks management for hierarchical optimization.

## Usage

In CMake:

```cmake
include(FetchContent)
FetchContent_Declare(
    hqp-tasks
    GIT_REPOSITORY "https://github.com/gianluca-garofalo/hqp-tasks.git"
    GIT_TAG "main"
)
FetchContent_MakeAvailable(hqp-tasks)

target_link_libraries(my_target PRIVATE hqp-tasks)
```
