import numpy as np

# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):
    if x <= 0 or y <= 0:
        return 0.0
    elif x >= 1 and y >= 1:
        return 1.0
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    else:  # 0 < x < 1 and 0 < y < 1
        return x * y

def rectangle_probability(x1, x2, y1, y2):
    return (joint_cdf_unit_square(x2, y2)
          - joint_cdf_unit_square(x1, y2)
          - joint_cdf_unit_square(x2, y1)
          + joint_cdf_unit_square(x1, y1))

def marginal_fx_unit_square(x):
    return 1.0 if 0 < x < 1 else 0.0

def marginal_fy_unit_square(y):
    return 1.0 if 0 < y < 1 else 0.0

# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

_joint_table = {
    (0, 0): 1/4,
    (0, 1): 1/4,
    (0, 2): 0.0,
    (1, 0): 0.0,
    (1, 1): 1/4,
    (1, 2): 1/4,
}

def joint_pmf_heads(x, y):
    return _joint_table.get((x, y), 0.0)

def marginal_px_heads(x):
    return sum(joint_pmf_heads(x, y) for y in range(3))

def marginal_py_heads(y):
    return sum(joint_pmf_heads(x, y) for x in range(2))

def check_independence_heads():
    for x in range(2):
        for y in range(3):
            if not np.isclose(joint_pmf_heads(x, y),
                              marginal_px_heads(x) * marginal_py_heads(y)):
                return False
    return True
