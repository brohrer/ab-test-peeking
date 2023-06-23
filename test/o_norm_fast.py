import numpy as np

N_STEPS = 10_000_000
N_RUNS = 10_000
N_TEST = 100
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
        sum_a = 0
        sum_b = 0
        sum_sq_a = 0
        sum_sq_b = 0

        for i_col in range(N_COLS):
            i_step = (i_col + 1) * N_TEST
            new_a = draw_from_A(N_TEST)
            new_b = draw_from_B(N_TEST)
            sum_a += np.sum(new_a)
            sum_b += np.sum(new_b)
            sum_sq_a += np.sum(new_a**2)
            sum_sq_b += np.sum(new_b**2)

            mean_a = sum_a / i_step
            mean_b = sum_b / i_step
            var_a = (sum_sq_a - i_step * mean_a ** 2) / i_step
            var_b = (sum_sq_b - i_step * mean_b ** 2) / i_step
            pooled_var = (var_a + var_b) / 2
            std_err = np.sqrt(2 * pooled_var / i_step)
            t_factor = 1.0
            t_stat = 1.96 * t_factor
            half_interval = std_err * t_stat
            difference = mean_b - mean_a
            upper_bound[i_run, i_col] = difference + half_interval
            lower_bound[i_run, i_col] = difference - half_interval

    # Save upper and lower bounds for analysis
    np.save("upper.npy", upper_bound)
    np.save("lower.npy", lower_bound)
    np.save("n_tests.npy", N)


def draw_from_A(n):
    return np.random.normal(loc=MEAN_A, scale=np.sqrt(VAR_AB), size=n)


def draw_from_B(n):
    return np.random.normal(loc=MEAN_B, scale=np.sqrt(VAR_AB), size=n)


main()
