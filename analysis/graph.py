import os
import click
import pandas as pd
import matplotlib.pyplot as plt


def plot(csv_filename: str):
    if not os.path.exists(csv_filename):
        raise FileExistsError(f"invalid csv filename:{csv_filename}")

    data = pd.read_csv(csv_filename)
    expected_keys = ["vertices", "kernel_time(ms)", "total_time(ms)"]
    for key in expected_keys:
        if key not in data.columns:
            raise KeyError(f"missing expected key:{key}")

    # plot data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(
        data["vertices"], data["kernel_time(ms)"], label="Kernel Time (ms)", marker="o"
    )
    # ax1.set_title("Kernel Time and Vertices")
    ax1.set_xlabel("Vertices")
    ax1.set_ylabel("Average mean curvature kernel time (ms)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(
        data["vertices"],
        data["total_time(ms)"],
        label="Total Time (ms)",
        marker="o",
        color="orange",
    )
    ax2.set_title("Total Time vs Vertices")
    ax2.set_xlabel("Vertices")
    ax2.set_ylabel("Total Time (ms)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    @click.command()
    @click.option(
        "--csv_filename",
        "-cf",
        default="utils/mean_curvature_01.csv",
        type=str,
    )
    def main(csv_filename: str):
        plot(csv_filename)

    main()
