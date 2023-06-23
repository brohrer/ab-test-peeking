import numpy as np
import matplotlib.pyplot as plt

N_STEPS = 1_000_000
N_RUNS = 1000
N_TEST = 100

# Create two distributions with the same variance but with different means.
CHANGE = 0.0
MEAN_A = 2
MEAN_B = MEAN_A * (1 + CHANGE)
VAR_AB = 1


def main():
    upper_bound = np.ones((N_RUNS, N_STEPS))
    lower_bound = -np.ones((N_RUNS, N_STEPS))

    upper_crossover = np.zeros(N_RUNS)
    lower_crossover = np.zeros(N_RUNS)
    crossover_count = 0

    for i_run in range(N_RUNS):
        print(f"Starting run {i_run}", end="\r")
        A = np.zeros(N_STEPS)
        B = np.zeros(N_STEPS)
        A[0] = draw_from_A()
        B[0] = draw_from_B()
        A[1] = draw_from_A()
        B[1] = draw_from_B()

        mean_a = np.zeros(N_STEPS)
        mean_b = np.zeros(N_STEPS)
        var_a = np.zeros(N_STEPS)
        var_b = np.zeros(N_STEPS)

        for i_step in range(2, N_STEPS):
            A[i_step] = draw_from_A()
            B[i_step] = draw_from_B()

            if i_step % N_TEST == 0:
                mean_a[i_step] = np.mean(A[:i_step])
                mean_b[i_step] = np.mean(B[:i_step])
                var_a[i_step] = np.var(A[:i_step])
                var_b[i_step] = np.var(B[:i_step])
                pooled_var = (var_a[i_step] + var_b[i_step]) / 2
                std_err = np.sqrt(2 * pooled_var / i_step)
                # t_stat = stats.t.ppf(.975, i_step * 2)
                t_factor = 1.0
                t_stat = 1.96 * t_factor
                half_interval = std_err * t_stat
                difference = mean_b[i_step] - mean_a[i_step]
                upper_bound[i_run, i_step]  = difference + half_interval
                lower_bound[i_run, i_step]  = difference - half_interval
            else:
                upper_bound[i_run, i_step] = upper_bound[i_run, i_step - 1]
                lower_bound[i_run, i_step] = lower_bound[i_run, i_step - 1]


    # Save upper and lower bounds for analysis
    np.save('upper.npy', upper_bound)
    np.save('lower.npy', lower_bound)


def draw_from_A():
    return np.random.normal(loc=MEAN_A, scale=np.sqrt(VAR_AB))


def draw_from_B():
    return np.random.normal(loc=MEAN_B, scale=np.sqrt(VAR_AB))


main()
