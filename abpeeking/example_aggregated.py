import numpy as np
import abpeeking.p_values as p_values

A = np.array([2, 3, 4, 5, 6])
B = np.array([4, 5, 6, 7])
count_a = A.size
count_b = B.size
mean_a = np.mean(A)
mean_b = np.mean(B)
var_a = np.var(A, ddof=1)
var_b = np.var(B, ddof=1)

print(
    p_values.calcuate_with_aggregates(
        count_a, count_b, mean_a, mean_b, var_a, var_b
    )
)
