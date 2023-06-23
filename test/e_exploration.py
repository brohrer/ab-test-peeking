import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import abpeeking.p_values as p_values

N_STEPS = 100_000
N_RUNS = 1
# N_FILT = 1000
N_TEST = 1
N_CONF_INTS = 8

# Create two distributions with the same variance but with different means.
CHANGE = 0.0
MEAN_A = 2
MEAN_B = MEAN_A * (1 + CHANGE)
VAR_AB = 1

# Set the style for the lines in the plots
ALPHA = 0.3
LINEWIDTH = 0.3
COLOR = "black"


def main():
    estimates_fig, mean_a_ax, mean_b_ax, mean_diff_ax = create_estimates_axes()
    # pvalues_fig, corrected_ax, uncorrected_ax = create_pvalues_axes()
    # (
    #     filt_pvalues_fig,
    #     corrected_filt_ax,
    #     uncorrected_filt_ax,
    # ) = create_filtered_pvalues_axes()

    std_errs = np.ones((N_RUNS, N_STEPS))

    # corrected_p_values = np.ones((N_RUNS, N_STEPS))
    # uncorrected_p_values = np.ones((N_RUNS, N_STEPS))
    # corrected_crossover = np.zeros(N_RUNS)
    # uncorrected_crossover = np.zeros(N_RUNS)

    # corrected_filt_p_values = np.ones((N_RUNS, N_STEPS))
    # uncorrected_filt_p_values = np.ones((N_RUNS, N_STEPS))
    # corrected_filt_crossover = np.zeros(N_RUNS)
    # uncorrected_filt_crossover = np.zeros(N_RUNS)

    # corrected_crossover_count = 0
    # uncorrected_crossover_count = 0
    # corrected_filt_crossover_count = 0
    # uncorrected_filt_crossover_count = 0

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

        for i_step in range(2, N_STEPS):
            A[i_step] = draw_from_A()
            B[i_step] = draw_from_B()
            mean_a[i_step] = np.mean(A[:i_step])
            mean_b[i_step] = np.mean(B[:i_step])

            if i_step % N_TEST == 0:
                std_errs[i_run, i_step] = stats.sem(
                    B[:i_step] - A[:i_step]  # , ddof=i_step * 2
                )
            else:
                std_errs[i_run, i_step] = std_errs[i_run, i_step - 1]

            # i_filt_start = np.maximum(0, i_step - N_FILT)
            # corrected_filt_p_values[i_run, i_step] = np.max(
            #     corrected_p_values[i_run, i_filt_start:i_step]
            #  )
            # uncorrected_filt_p_values[i_run, i_step] = np.max(
            #     uncorrected_p_values[i_run, i_filt_start:i_step]
            # )

            # all_mean_a[i_step] = np.mean(A[: i_step + 1])
            # all_mean_b[i_step] = np.mean(B[: i_step + 1])

        """
        try:
            corrected_crossover[i_run] = np.min(
                np.where(corrected_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            corrected_crossover[i_run] = N_STEPS - 1
            corrected_crossover_count += 1

        try:
            uncorrected_crossover[i_run] = np.min(
                np.where(uncorrected_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            uncorrected_crossover[i_run] = N_STEPS - 1
            uncorrected_crossover_count += 1

        try:
            corrected_filt_crossover[i_run] = np.min(
                np.where(corrected_filt_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            corrected_filt_crossover[i_run] = N_STEPS - 1
            corrected_filt_crossover_count += 1

        try:
            uncorrected_filt_crossover[i_run] = np.min(
                np.where(uncorrected_filt_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            uncorrected_filt_crossover[i_run] = N_STEPS - 1
            uncorrected_filt_crossover_count += 1
        """

        """
        corrected_ax.plot(
            [corrected_crossover[i_run], corrected_crossover[i_run]],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )
        uncorrected_ax.plot(
            [uncorrected_crossover[i_run], uncorrected_crossover[i_run]],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )

        corrected_filt_ax.plot(
            [corrected_filt_crossover[i_run], corrected_filt_crossover[i_run]],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )
        uncorrected_filt_ax.plot(
            [
                uncorrected_filt_crossover[i_run],
                uncorrected_filt_crossover[i_run],
            ],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )
        """

        # Plot the means as they evolve
        # mean_diff = all_mean_b - all_mean_a

        mean_a_ax.plot(mean_a, color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH)
        mean_b_ax.plot(mean_b, color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH)
        mean_diff = mean_b - mean_a
        zs = [1.96, 1.43, 1.15, 0.675, -0.675, -1.15, -1.43, -1.96]
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[0], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[1], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[2], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[3], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[4], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[5], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[6], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff  + std_errs[i_run, :] * zs[7], color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            [0, N_STEPS],
            [MEAN_B - MEAN_A, MEAN_B - MEAN_A],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )

    """
    print(
        "corrected crossover",
        "triggered pct", 100 * (1.0 - float(corrected_crossover_count) / float(N_RUNS)),
        "mean", np.mean(corrected_crossover),
        "median", np.median(corrected_crossover),
        "stddev", np.sqrt(np.var(corrected_crossover)),
    )
    print(
        "uncorrected crossover",
        "triggered pct", 100 * (1.0 - float(uncorrected_crossover_count) / float(N_RUNS)),
        "mean", np.mean(uncorrected_crossover),
        "median", np.median(uncorrected_crossover),
        "stddev", np.sqrt(np.var(uncorrected_crossover)),
    )
    print(
        "corrected filt crossover",
        "triggered pct", 100 * (1.0 - float(corrected_filt_crossover_count) / float(N_RUNS)),
        "mean", np.mean(corrected_filt_crossover),
        "median", np.median(corrected_filt_crossover),
        "steddev", np.sqrt(np.var(corrected_filt_crossover)),
    )
    print(
        "uncorrected filt crossover",
        "triggered pct", 100 * (1.0 - float(uncorrected_filt_crossover_count) / float(N_RUNS)),
        "mean", np.mean(uncorrected_filt_crossover),
        "median", np.median(uncorrected_filt_crossover),
        "stddev", np.sqrt(np.var(uncorrected_filt_crossover)),
    )
    """
    """
    corrected_ax.plot(
        corrected_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )
    uncorrected_ax.plot(
        uncorrected_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )

    corrected_filt_ax.plot(
        corrected_filt_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )
    uncorrected_filt_ax.plot(
        uncorrected_filt_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )
    """

    plt.show()


def draw_from_A():
    return np.random.normal(loc=MEAN_A, scale=np.sqrt(VAR_AB))


def draw_from_B():
    return np.random.normal(loc=MEAN_B, scale=np.sqrt(VAR_AB))


def create_estimates_axes():
    # Layout the axes within the figure
    fig_height = 8  # inches
    fig_width = 10  # inches

    x_spacing = 1  # inches
    y_spacing = 0.5  # inches

    x_min = x_spacing / fig_width
    x_max = 1 - x_min
    dx_ax = x_max - x_min

    y_border = y_spacing / fig_height
    dy_ax = (1 - 4 * y_border) / 3
    fig = plt.figure("estimates", figsize=(fig_width, fig_height))

    y_min = y_border
    mean_diff_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    y_min = 2 * y_border + dy_ax
    mean_b_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    y_min = 3 * y_border + 2 * dy_ax
    mean_a_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    mean_a_ax.set_xlim(0, N_STEPS)
    mean_b_ax.set_xlim(0, N_STEPS)
    mean_diff_ax.set_xlim(0, N_STEPS)

    y_scale = 0.1
    mean_a_ax.set_ylim(MEAN_A - y_scale, MEAN_A + y_scale)
    mean_b_ax.set_ylim(MEAN_B - y_scale, MEAN_B + y_scale)
    mean_diff_ax.set_ylim(MEAN_B - MEAN_A - y_scale, MEAN_B - MEAN_A + y_scale)

    mean_a_ax.set_ylabel("Average value of group A")
    mean_b_ax.set_ylabel("Average value of group B")
    mean_diff_ax.set_ylabel("Difference between group averages")

    mean_diff_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing actual values
    dashed_line_color = "white"
    dashed_linewidth = 1
    mean_a_ax.plot(
        [0, N_STEPS],
        [MEAN_A, MEAN_A],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    mean_b_ax.plot(
        [0, N_STEPS],
        [MEAN_B, MEAN_B],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    mean_diff_ax.plot(
        [0, N_STEPS],
        [MEAN_B - MEAN_A, MEAN_B - MEAN_A],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )

    return fig, mean_a_ax, mean_b_ax, mean_diff_ax


"""
def create_pvalues_axes():
    # Layout the axes within the figure
    fig_height = 8  # inches
    fig_width = 10  # inches

    x_spacing = 1  # inches
    y_spacing = 0.5  # inches

    x_min = x_spacing / fig_width
    x_max = 1 - x_min
    dx_ax = x_max - x_min

    y_border = y_spacing / fig_height
    dy_ax = (1 - 3 * y_border) / 2
    fig = plt.figure("p-values", figsize=(fig_width, fig_height))

    y_min = y_border
    p_uncorrected_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    y_min = 2 * y_border + dy_ax
    p_corrected_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    p_corrected_ax.set_xlim(0, N_STEPS)
    p_uncorrected_ax.set_xlim(0, N_STEPS)

    p_corrected_ax.set_ylim(-0.06, 1.01)
    p_uncorrected_ax.set_ylim(-0.06, 1.01)

    p_corrected_ax.set_ylabel("Corrected p-values")
    p_uncorrected_ax.set_ylabel("Uncorrected (standard) p-values")

    p_uncorrected_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing significance thresholds
    dashed_line_color = "blue"
    dashed_linewidth = 1
    p_corrected_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    p_uncorrected_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    return fig, p_corrected_ax, p_uncorrected_ax


def create_filtered_pvalues_axes():
    # Layout the axes within the figure
    fig_height = 8  # inches
    fig_width = 10  # inches

    x_spacing = 1  # inches
    y_spacing = 0.5  # inches

    x_min = x_spacing / fig_width
    x_max = 1 - x_min
    dx_ax = x_max - x_min

    y_border = y_spacing / fig_height
    dy_ax = (1 - 3 * y_border) / 2
    fig = plt.figure("filtered p-values", figsize=(fig_width, fig_height))

    y_min = y_border
    p_uncorrected_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    y_min = 2 * y_border + dy_ax
    p_corrected_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    p_corrected_ax.set_xlim(0, N_STEPS)
    p_uncorrected_ax.set_xlim(0, N_STEPS)

    p_corrected_ax.set_ylim(-0.06, 1.01)
    p_uncorrected_ax.set_ylim(-0.06, 1.01)

    p_corrected_ax.set_ylabel("Corrected p-values")
    p_uncorrected_ax.set_ylabel("Uncorrected (standard) p-values")

    p_uncorrected_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing significance thresholds
    dashed_line_color = "blue"
    dashed_linewidth = 1
    p_corrected_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    p_uncorrected_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    return fig, p_corrected_ax, p_uncorrected_ax
"""


main()
