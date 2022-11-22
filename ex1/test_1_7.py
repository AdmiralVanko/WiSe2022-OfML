from collections import namedtuple

import exercise_1 as student


def run_test():
    seam_carving_func = student.seam_carving
    if hasattr(student, "dynamic_programming"):
        dynamic_programming_func = student.dynamic_programming

    backtrack_func = student.backtrack
    if hasattr(student, "backtrack_tree"):
        backtrack_func = student.backtrack_tree

    seam_carving_func("tower.jpg")

    print("Ok")


if __name__ == "__main__":
    run_test()
