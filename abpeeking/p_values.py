import numpy as np


def calculate(A, B):
    A = np.array(A)
    B = np.array(B)

    count_a = A.size
    count_b = B.size
    mean_a = np.mean(A)
    mean_b = np.mean(B)

    var_a = np.sum((A - mean_a)**2) / (count_a - 1)
    var_b = np.sum((B - mean_b)**2) / (count_b - 1)

    p_value = calcuate_with_aggregates(count_a, count_b, mean_a, mean_b, var_a, var_b)
    return p_value


def calcuate_with_aggregates(count_a, count_b, mean_a, mean_b, var_a, var_b):
    # pooled variance
    var_p = (
        ( (count_a - 1) * var_a + (count_b - 1) * var_b) /
        ( (count_a - 1) + (count_b - 1) )
    )

    n = count_a + count_b

    # A small non-zero amount to keep everything non-zero and prevent overflow)
    epsilon = 1e-2

    # Equation 11 from the original paper
    # http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p1517.pdf
    Lambda = (
        np.sqrt( (2 * var_p) / (2 * var_p + n) ) *
        np.exp(
            n ** 2 * (mean_a - mean_b) ** 2 /
            (4 * (var_p + epsilon) * (2 * var_p + n) )
        )
    )
    p_value = 1 / Lambda

    # Put a ceiling on the p-value at 1. It is a probability, after all.
    p_value = np.minimum(p_value, 1)

    return p_value
