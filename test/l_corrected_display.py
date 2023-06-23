import numpy as np
import matplotlib.pyplot as plt

# Set the style for the lines in the plots
PLOTTING = True
ALPHA = 0.3
LINEWIDTH = 0.3
COLOR = "black"


def main():
    make_ci_violations_plot()


def make_ci_violations_plot():

    upper_bound = np.load('upper.npy')
    lower_bound = np.load('lower.npy')
    N_RUNS, N_STEPS = upper_bound.shape

    se = (upper_bound - lower_bound) / (2 * 1.96)
    correction_factor = 1.234
    upper_bound += se * correction_factor
    lower_bound -= se * correction_factor
    upper_violations = upper_bound <= 0
    lower_violations = lower_bound >= 0
    cumulative_uv = np.cumsum(upper_violations, axis=1)
    cumulative_lv = np.cumsum(lower_violations, axis=1)
    cumulative_violations = cumulative_uv + cumulative_lv
    violation_history = np.minimum(cumulative_violations, 1)
    cumulative_v_by_step = np.sum(violation_history, axis=0)
    fraction_violations = cumulative_v_by_step / float(N_RUNS)

    fig, ax = create_violations_axes(N_STEPS)
    ax.plot(fraction_violations)
    ax.set_ylim(0, np.max(fraction_violations) * 1.05)

    plt.show()


def make_crossover_point_plot():
    estimates_fig, mean_diff_ax = create_estimates_axes()

    upper_crossover = np.zeros(N_RUNS)
    lower_crossover = np.zeros(N_RUNS)
    crossover_count = 0

    for i_run in range(N_RUNS):

        try:
            upper_crossover[i_run] = CROSSOVER_FLOOR + np.min(np.where(upper_bound[i_run, CROSSOVER_FLOOR:] <= 0)[0])
        except ValueError:
            upper_crossover[i_run] = N_STEPS - 1

        try:
            lower_crossover[i_run] = CROSSOVER_FLOOR + np.min(np.where(lower_bound[i_run, CROSSOVER_FLOOR:] >= 0)[0])
        except ValueError:
            lower_crossover[i_run] = N_STEPS - 1

        if PLOTTING:
            mean_diff_ax.plot(
                upper_bound[i_run, :],
                color=COLOR,
                alpha=ALPHA,
                linewidth=LINEWIDTH,
            )
            mean_diff_ax.plot(
                lower_bound[i_run, :],
                color=COLOR,
                alpha=ALPHA,
                linewidth=LINEWIDTH,
            )

    i_crossover = np.minimum(upper_crossover, lower_crossover)
    i_crossover = i_crossover[np.where(i_crossover < (N_STEPS - 1))]
    print("                                    ")
    print(i_crossover)
    print(
        "corrected crossover",
        "triggered pct", 100 * (float(i_crossover.size) / float(N_RUNS)),
        "mean", np.mean(i_crossover),
        "median", np.median(i_crossover),
        "stddev", np.sqrt(np.var(i_crossover)),
    )
    print('crossovers at')
    print(i_crossover)

    if PLOTTING:
        plt.show()


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
    dy_ax = 1 - 2 * y_border
    fig = plt.figure("estimates", figsize=(fig_width, fig_height))

    y_min = y_border
    mean_diff_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    mean_diff_ax.set_xlim(0, N_STEPS)

    y_scale = 0.1
    mean_diff_ax.set_ylim(- y_scale, y_scale)

    mean_diff_ax.set_ylabel("Difference between group averages")

    mean_diff_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing actual values
    dashed_line_color = "darkblue"
    dashed_linewidth = 1
    mean_diff_ax.plot(
        [0, N_STEPS],
        [0, 0],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )

    return fig, mean_diff_ax


def create_violations_axes(N_STEPS):
    # Layout the axes within the figure
    fig_height = 8  # inches
    fig_width = 10  # inches

    x_spacing = 1  # inches
    y_spacing = 0.5  # inches

    x_min = x_spacing / fig_width
    x_max = 1 - x_min
    dx_ax = x_max - x_min

    y_border = y_spacing / fig_height
    dy_ax = 1 - 2 * y_border
    fig = plt.figure("estimates", figsize=(fig_width, fig_height))

    y_min = y_border
    violations_ax = fig.add_axes((x_min, y_min, dx_ax, dy_ax))

    # Format the axes
    violations_ax.set_xlim(0, N_STEPS)

    violations_ax.set_ylabel("Fraction of run violating 95% conf int")

    violations_ax.set_xlabel("Number of samples collected from each group")

    # Add lines showing actual values
    dashed_line_color = "darkblue"
    dashed_linewidth = 1
    violations_ax.plot(
        [0, N_STEPS],
        [0.05, 0.05],
        color=dashed_line_color,
        linestyle="dashed",
        linewidth=dashed_linewidth,
        zorder=2,
    )

    return fig, violations_ax


main()
