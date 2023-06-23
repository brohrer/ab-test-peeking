import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import abpeeking.p_values as p_values

N_STEPS = 3000
N_RUNS = 200

# Create two distributions with the same variance but with different means.
VAR_AB = 1
MEAN_A = 2
MEAN_B = 2.1

# Set the style for the lines in the plots
ALPHA = 0.3
LINEWIDTH = 0.3
COLOR = "black"


def main():
    estimates_fig, mean_a_ax, mean_b_ax, mean_diff_ax = create_estimates_axes()
    pvalues_fig, corrected_ax, uncorrected_ax = create_pvalues_axes()

    corrected_p_values = np.ones((N_RUNS, N_STEPS))
    uncorrected_p_values = np.ones((N_RUNS, N_STEPS))
    corrected_crossover = np.zeros(N_RUNS)
    uncorrected_crossover = np.zeros(N_RUNS)

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

            # corrected_p_values[i_run, i_step] = p_values.calculate(
            #     A[: i_step + 1], B[: i_step + 1]
            # )
            # uncorrected_p_values[i_run, i_step] = stats.ttest_ind(
            #     A[: i_step + 1], B[: i_step + 1]
            # ).pvalue
            corrected_p_values[i_run, i_step] = np.minimum(
                p_values.calculate(A[: i_step + 1], B[: i_step + 1]),
                corrected_p_values[i_run, i_step - 1])
            uncorrected_p_values[i_run, i_step] = np.minimum(
                stats.ttest_ind(A[: i_step + 1], B[: i_step + 1]).pvalue,
                uncorrected_p_values[i_run, i_step - 1])

            all_mean_a[i_step] = np.mean(A[: i_step + 1])
            all_mean_b[i_step] = np.mean(B[: i_step + 1])

        try:
            corrected_crossover[i_run] = np.min(
                np.where(corrected_p_values[i_run, :] < .05)[0])
        except ValueError:
            corrected_crossover[i_run] = N_STEPS - 1

        try:
            uncorrected_crossover[i_run] = np.min(
                np.where(uncorrected_p_values[i_run, :] < .05)[0])
        except ValueError:
            uncorrected_crossover[i_run] = N_STEPS - 1

        corrected_ax.plot(
            [corrected_crossover[i_run], corrected_crossover[i_run]],
            [0.0, -0.05],
            color='blue',
            linewidth=.5,
            alpha=.7,
        )
        uncorrected_ax.plot(
            [uncorrected_crossover[i_run], uncorrected_crossover[i_run]],
            [0.0, -0.05],
            color='blue',
            linewidth=.5,
            alpha=.7,
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

    corrected_ax.plot(
        corrected_p_values.transpose(), color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
    )
    uncorrected_ax.plot(
        uncorrected_p_values.transpose(), color=COLOR, alpha=ALPHA, linewidth=LINEWIDTH
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

    mean_a_ax.set_ylim(1.9, 2.1)
    mean_b_ax.set_ylim(2.0, 2.2)
    mean_diff_ax.set_ylim(0.0, 0.2)

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

    p_uncorrected_ax.set_xlabel(
        "Number of samples collected from each group"
    )

    # Add lines showing significance thresholds
    dashed_line_color = 'blue'
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


main()