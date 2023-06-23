import numpy as np
import matplotlib.pyplot as plt

N_STEPS = 1_000_000
N_RUNS = 1000
N_TEST = 1000
N_COLS = N_STEPS // N_TEST

# Create two distributions with the same variance but with different means.
CHANGE = 0.0
MEAN_A = 2
MEAN_B = MEAN_A * (1 + CHANGE)
VAR_AB = 1


def main():
    upper_bound = np.ones((N_RUNS, N_COLS))
    lower_bound = -np.ones((N_RUNS, N_COLS))
    N = np.arange(0, N_STEPS, N_TEST)

    for i_run in range(N_RUNS):
        print(f"Starting run {i_run}", end="\r")
        A = np.zeros(N_STEPS)
        B = np.zeros(N_STEPS)

        mean_a = np.zeros(N_COLS)
        mean_b = np.zeros(N_COLS)
        var_a = np.zeros(N_COLS)
        var_b = np.zeros(N_COLS)

        for i_col in range(N_COLS):
            i_step_last = i_col * N_TEST
            i_step = (i_col + 1) * N_TEST
            A[i_step_last : i_step] = draw_from_A(N_TEST)
            B[i_step_last : i_step] = draw_from_B(N_TEST)

            mean_a[i_col] = np.mean(A[:i_step])
            mean_b[i_col] = np.mean(B[:i_step])
            var_a[i_col] = np.var(A[:i_step])
            var_b[i_col] = np.var(B[:i_step])
            pooled_var = (var_a[i_col] + var_b[i_col]) / 2
            std_err = np.sqrt(2 * pooled_var / i_step)
            # t_stat = stats.t.ppf(.975, i_step * 2)
            t_factor = 1.0
            t_stat = 1.96 * t_factor
            half_interval = std_err * t_stat
            difference = mean_b[i_col] - mean_a[i_col]
            upper_bound[i_run, i_col]  = difference + half_interval
            lower_bound[i_run, i_col]  = difference - half_interval


    # Save upper and lower bounds for analysis
    np.save('upper.npy', upper_bound)
    np.save('lower.npy', lower_bound)
    np.save('n_tests.npy',N)


def draw_from_A(n):
    return np.random.normal(loc=MEAN_A, scale=np.sqrt(VAR_AB), size=n)


def draw_from_B(n):
    return np.random.normal(loc=MEAN_B, scale=np.sqrt(VAR_AB), size=n)


main()
