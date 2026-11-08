import numpy as np
import pyhqp


def main():
    solver = pyhqp.HierarchicalQP(3, 2)

    # Higher priority: x[0] == 0
    # Lower priority: x[1] == 8
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    bl = np.array([0.0, 8.0, -np.inf])
    bu = np.array([0.0, np.inf, 8.0])
    breaks = np.array([1, 3], dtype=np.int32)

    solver.set_problem(A, bl, bu, breaks)
    solution = solver.get_primal()
    print(f"Status: solved = {solution is not None}")
    print(f"Solution: {solution}")


if __name__ == "__main__":
    main()
