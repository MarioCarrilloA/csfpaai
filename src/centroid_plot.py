import json
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np

out_path = "../res/centroids_charts/"

# Classes labels CIFAR-10
classes = ('plane',
           'auto',
           'bird',
           'cat',
           'deer',
           'dog',
           'frog',
           'horse',
           'ship',
           'truck'
)

hist_colors = (
        'red',
        'blue',
        'olivedrab',
        'navy',
        'green',
        'gold',
        'steelblue',
        'violet',
        'sienna',
        'purple'
)

# Fixing random state for reproducibility
np.random.seed(19680801)

# some random data
x = np.random.randn(1000)
y = np.random.randn(1000)


def scatter_hist(class_name, x, y, ax, ax_histx, ax_histy, color):
    # no labels
    #ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="x")
    #ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="y")

    # the scatter plot:
    ax.scatter(x, y, color=color, alpha=0.5, s=30, label=class_name)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, histtype='step', color=color)
    ax_histy.hist(y, bins=bins, orientation='horizontal',histtype='step', color=color)


def plot_charts(data_file, basename):
    # start with a square Figure
    plt.clf()
    fig = plt.figure(figsize=(20, 20))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim([0, 32])
    ax.set_ylim([32, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    data_dict = data_file[0]
    for i in range(len(classes)):
        points = data_dict[str(i)]
        zip_points = list(zip(*points))
        x = zip_points[0]
        y = zip_points[1]
        color = hist_colors[i]

        class_name = classes[i]
        scatter_hist(class_name, x, y, ax, ax_histx, ax_histy, color)

    ax.legend()

    fig.suptitle(basename, fontsize=26)
    plt.savefig(out_path + "centroids_data_{}.jpg".format(basename), bbox_inches='tight')



def main():
    if len(sys.argv) <= 1:
        print("error: no input file")
        sys.exit(0)

    input_file = sys.argv[1]
    if os.path.isdir(out_path) == False:
        os.makedirs(out_path)

    tokens = input_file.split("_")
    tokens = tokens[-1].split(".")
    basename = tokens[0]
    data_file = []
    if os.path.isfile(input_file) == True:
        with open(input_file, 'r') as json_file:
            data_file = json.load(json_file)
            result_id = len(data_file) + 1
            json_file.close()
    else:
        print("The file {} does not exist".format(input_file))

    print("plotting...")
    plot_charts(data_file, basename)
    print("Done!")

if __name__ == "__main__":
    main()
