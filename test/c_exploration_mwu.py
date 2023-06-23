import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import abpeeking.p_values as p_values

N_STEPS = 2_000
N_RUNS = 3
N_FILT = 1
N_TEST = 1

# Create two distributions with the same variance but with different means.
CHANGE = .1
MEAN_A = 2
MEAN_B = MEAN_A * (1 + CHANGE)
VAR_AB = 1

# Set the style for the lines in the plots
ALPHA = 0.3
LINEWIDTH = 0.3
COLOR = "black"


def main():
    estimates_fig, mean_a_ax, mean_b_ax, mean_diff_ax = create_estimates_axes()
    pvalues_fig, mannwhitney_ax, ttest_ax = create_pvalues_axes()
    (
        filt_pvalues_fig,
        mannwhitney_filt_ax,
        ttest_filt_ax,
    ) = create_filtered_pvalues_axes()

    mannwhitney_p_values = np.ones((N_RUNS, N_STEPS))
    ttest_p_values = np.ones((N_RUNS, N_STEPS))
    mannwhitney_crossover = np.zeros(N_RUNS)
    ttest_crossover = np.zeros(N_RUNS)

    mannwhitney_filt_p_values = np.ones((N_RUNS, N_STEPS))
    ttest_filt_p_values = np.ones((N_RUNS, N_STEPS))
    mannwhitney_filt_crossover = np.zeros(N_RUNS)
    ttest_filt_crossover = np.zeros(N_RUNS)

    for i_run in range(N_RUNS):
        print(f"Starting run {i_run}", end="\r")
        A = np.zeros(N_STEPS)
        B = np.zeros(N_STEPS)
        A[0] = draw_from_A()
        B[0] = draw_from_B()

        all_mean_a = np.zeros(N_STEPS)
        all_mean_b = np.zeros(N_STEPS)

        for i_step in range(1, N_STEPS):
            A[i_step] = draw_from_A()
            B[i_step] = draw_from_B()

            if i_step % N_TEST == 0:
                # mannwhitney_p_values[i_run, i_step] = p_values.calculate(
                #     A[: i_step + 1], B[: i_step + 1]
                # )
                mannwhitney_p_values[i_run, i_step] = stats.mannwhitneyu(
                    A[: i_step + 1], B[: i_step + 1]
                ).pvalue
                ttest_p_values[i_run, i_step] = stats.ttest_ind(
                    A[: i_step + 1], B[: i_step + 1]
                ).pvalue
            else:
                # mannwhitney_p_values[i_run, i_step] = mannwhitney_p_values[
                #     i_run, i_step - 1]
                mannwhitney_p_values[i_run, i_step] = mannwhitney_p_values[
                    i_run, i_step - 1]
                ttest_p_values[i_run, i_step] = ttest_p_values[
                    i_run, i_step - 1]

            i_filt_start = np.maximum(0, i_step - N_FILT)
            mannwhitney_filt_p_values[i_run, i_step] = np.max(
                mannwhitney_p_values[i_run, i_filt_start:i_step]
            )
            ttest_filt_p_values[i_run, i_step] = np.max(
                ttest_p_values[i_run, i_filt_start:i_step]
            )

            all_mean_a[i_step] = np.mean(A[: i_step + 1])
            all_mean_b[i_step] = np.mean(B[: i_step + 1])

        try:
            mannwhitney_crossover[i_run] = np.min(
                np.where(mannwhitney_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            mannwhitney_crossover[i_run] = N_STEPS - 1

        try:
            ttest_crossover[i_run] = np.min(
                np.where(ttest_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            ttest_crossover[i_run] = N_STEPS - 1

        try:
            mannwhitney_filt_crossover[i_run] = np.min(
                np.where(mannwhitney_filt_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            mannwhitney_filt_crossover[i_run] = N_STEPS - 1

        try:
            ttest_filt_crossover[i_run] = np.min(
                np.where(ttest_filt_p_values[i_run, :] < 0.05)[0]
            )
        except ValueError:
            ttest_filt_crossover[i_run] = N_STEPS - 1

        mannwhitney_ax.plot(
            [mannwhitney_crossover[i_run], mannwhitney_crossover[i_run]],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )
        ttest_ax.plot(
            [ttest_crossover[i_run], ttest_crossover[i_run]],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )

        mannwhitney_filt_ax.plot(
            [mannwhitney_filt_crossover[i_run], mannwhitney_filt_crossover[i_run]],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )
        ttest_filt_ax.plot(
            [
                ttest_filt_crossover[i_run],
                ttest_filt_crossover[i_run],
            ],
            [0.0, -0.05],
            color="blue",
            linewidth=0.5,
            alpha=0.3,
        )

        # Plot the means as they evolve
        mean_diff = all_mean_b - all_mean_a

        mean_a_ax.plot(
            all_mean_a, color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_b_ax.plot(
            all_mean_b, color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )
        mean_diff_ax.plot(
            mean_diff, color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
        )

    print(
        "mannwhitney crossover",
        np.mean(mannwhitney_crossover),
        np.median(mannwhitney_crossover))
    print(
        "ttest crossover",
        np.mean(ttest_crossover),
        np.median(ttest_crossover))
    print(
        "mannwhitney filt crossover",
        np.mean(mannwhitney_filt_crossover),
        np.median(mannwhitney_filt_crossover))
    print(
        "ttest filt crossover",
        np.mean(ttest_filt_crossover),
        np.median(ttest_filt_crossover))


    mannwhitney_ax.plot(
        mannwhitney_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )
    ttest_ax.plot(
        ttest_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )

    mannwhitney_filt_ax.plot(
        mannwhitney_filt_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )
    ttest_filt_ax.plot(
        ttest_filt_p_values.transpose(),
        color=COLOR,
        alpha=ALPHA,
        linewidth=LINEWIDTH,
    )

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

    y_scale = .1
    # mean_a_ax.set_ylim(1.9, 2.1)
    # mean_b_ax.set_ylim(2.0, 2.2)
    # mean_diff_ax.set_ylim(0.0, 0.2)
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
    p_ttest_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    y_min = 2 * y_border + dy_ax
    p_mannwhitney_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    p_mannwhitney_ax.set_xlim(0, N_STEPS)
    p_ttest_ax.set_xlim(0, N_STEPS)

    p_mannwhitney_ax.set_ylim(-0.06, 1.01)
    p_ttest_ax.set_ylim(-0.06, 1.01)

    p_mannwhitney_ax.set_ylabel("Corrected p-values")
    p_ttest_ax.set_ylabel("Unmannwhitney (standard) p-values")

    p_ttest_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing significance thresholds
    dashed_line_color = "blue"
    dashed_linewidth = 1
    p_mannwhitney_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    p_ttest_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    return fig, p_mannwhitney_ax, p_ttest_ax


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
    p_ttest_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    y_min = 2 * y_border + dy_ax
    p_mannwhitney_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    p_mannwhitney_ax.set_xlim(0, N_STEPS)
    p_ttest_ax.set_xlim(0, N_STEPS)

    p_mannwhitney_ax.set_ylim(-0.06, 1.01)
    p_ttest_ax.set_ylim(-0.06, 1.01)

    p_mannwhitney_ax.set_ylabel("Corrected p-values")
    p_ttest_ax.set_ylabel("Unmannwhitney (standard) p-values")

    p_ttest_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing significance thresholds
    dashed_line_color = "blue"
    dashed_linewidth = 1
    p_mannwhitney_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    p_ttest_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )
    return fig, p_mannwhitney_ax, p_ttest_ax


main()
