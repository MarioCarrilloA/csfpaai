import os.path
import json
import matplotlib.pyplot as plt
import sys
import random
import statistics as stat
import numpy as np

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
out_path = "../res/charts/"
IMAGE_SIZE=(32, 32)


bar_colors = (
        'violet',
        'yellowgreen',
        'navajowhite',
        'tomato',
        'mediumorchid',
        'silver',
        'dodgerblue',
        'mediumpurple',
        'turquoise',
        'limegreen'
)


def plot_classes_results(data_file, dataset_type):
    dataset_key = ''
    if dataset_type == 'test':
        dataset_key = 'testds_classes_pcts'
    elif dataset_type == 'train':
        dataset_key = 'trainds_classes_pcts'
    else:
        dataset_key = 'testds_classes_pcts_ext'

    # Extract relevant data from JSON file
    x = []
    percentages_per_itr = []
    std_per_itr = []
    iterations = len(data_file)
    for itr in range(iterations):
        subvalues = []
        subiterations = len(data_file[itr])
        for subitr in range(subiterations):
            if (dataset_key in data_file[itr][subitr]) == False:
                print("{} key no found. It won't be plotted".format(dataset_key))
                return False
            # Collect the classes percentages per interations
            subvalues.append(data_file[itr][subitr][dataset_key])
        percentages_per_itr.append(subvalues)


    # compute mean and std of iterations per class
    for c in range(len(classes)):
        x = []
        avg_per_itr = []
        std_per_itr = []
        for pitr in range(len(percentages_per_itr)):
            subvalues = []
            x.append(pitr + 1)
            for subitr in range(len(percentages_per_itr[pitr])):
                subvalues.append(percentages_per_itr[pitr][subitr][c])
            avg_per_itr.append(stat.mean(subvalues))
            std_per_itr.append(stat.stdev(subvalues))

        # Create charts
        plt.clf()
        plt.margins(x=0)
        plt.xticks(x)
        plt.title("Using *{}* dataset - Class accuracy: ".format(dataset_type) +
                classes[c] + "\n", fontsize=15)
        plt.ylim([0,100])
        rgb = (random.random(), random.random(), random.random())
        #plt.bar(x, avg_per_itr,  yerr=std_per_itr, align='center', color=[rgb], capsize=10)
        plt.bar(x, avg_per_itr,  yerr=std_per_itr, align='center', color=bar_colors[c],
                    capsize=10, alpha=0.75)
        plt.ylabel("Percentage")
        plt.xlabel("Iterations")

        # Labels above bars
#        for i, v in enumerate(avg_per_itr):
#            plt.text(i + 0.7, v + std_per_itr[i] + 1, str(round(v, 2)) + "%")
        plt.savefig(out_path + "{}_".format(dataset_type) +
                classes[c] + ".jpg", bbox_inches='tight')


def plot_model_accuracy(data_file, dataset_type):
    dataset_key = ''
    if dataset_type == 'test':
        dataset_key = 'testds_accuracy'
    elif dataset_type == 'train':
        dataset_key = 'trainds_accuracy'
    else:
        dataset_key = 'testds_accuracy_ext'

    x = []
    accuracy_per_itr = []
    std_per_itr = []
    iterations = len(data_file)
    for itr in range(iterations):
        subvalues = []
        subiterations = len(data_file[itr])
        for subitr in range(subiterations):
            if (dataset_key in data_file[itr][subitr]) == False:
                print("{} key no found. It won't be plotted".format(dataset_key))
                return False
            subvalues.append(data_file[itr][subitr][dataset_key])
        accuracy_per_itr.append(stat.mean(subvalues))
        std_per_itr.append(stat.stdev(subvalues))
        x.append(itr + 1)

    rgb = (random.random(), random.random(), random.random())
    plt.clf()
    #plt.figure(figsize=(20, 10))
    plt.margins(x=0)
    plt.xticks(x)
    plt.title("Model accuracy per iteration " +
            "using *{}* dataset for testing".format(dataset_type) + "\n")
    plt.ylim([0,100])
    #plt.bar(x, accuracy_per_itr, yerr=std_per_itr, align='center', color=[rgb], capsize=10)
    plt.bar(x, accuracy_per_itr, yerr=std_per_itr, align='center', color='royalblue', capsize=10)
    plt.ylabel("Percentage")
    plt.xlabel("Iterations")

    # Labels above bars
    #for i, v in enumerate(accuracy_per_itr):
    #    plt.text(i + 0.7, v + std_per_itr[i] + 1, str(round(v, 2)) + "%")

    plt.savefig(out_path + "{}_".format(dataset_type) +
            "model_accuracy.jpg", bbox_inches='tight')


def plot_loss(data_file, dataset_type):
    dataset_key = ''
    if dataset_type == 'test':
        dataset_key = 'testds_loss'
    elif dataset_type == 'train':
        dataset_key = 'trainds_loss'
    else:
        dataset_key = 'testds_loss_ext'

    iterations = len(data_file)
    for itr in range(iterations):
        subvalues_train = []
        subvalues_test = []
        subiterations = len(data_file[itr])
        train_loss_avg = []
        train_loss_std = []
        test_loss_avg = []
        test_loss_std = []
        for subitr in range(subiterations):
            if (dataset_key in data_file[itr][subitr]) == False:
                print("{} key no found. It won't be plotted".format(dataset_key))
                return False
            subvalues_train.append(data_file[itr][subitr]['train_model_loss'])
            subvalues_test.append(data_file[itr][subitr][dataset_key])

        # Compute avg and std
        epochs = len(subvalues_train[0])
        x = []
        for epoch in range(len(subvalues_train[0])):
            train_epoch = []
            test_epoch = []
            for i in range(len(subvalues_train)):
                train_epoch.append(subvalues_train[i][epoch])
                test_epoch.append(subvalues_test[i][epoch])
            train_loss_avg.append(stat.mean(train_epoch))
            train_loss_std.append(stat.stdev(train_epoch))
            test_loss_avg.append(stat.mean(test_epoch))
            test_loss_std.append(stat.stdev(test_epoch))
            x.append(epoch+1)

        img_name = "loss_iteration_{}.png".format(itr)
        plt.clf()
        #plt.figure(figsize=(20, 10))
        plt.title("Train and test loss (with *{}* datset) ".format(dataset_type) +
            "- Iteration {}\n".format(itr))
        # Train
        y = np.array(train_loss_avg)
        error = np.array(train_loss_std)
        plt.errorbar(x, y=y,  color='#3385ff', label='Train loss')
        plt.fill_between(x, y-error, y+error,
                    alpha=0.5,  facecolor='gray', label='Train stdev')

        # Test
        y = np.array(test_loss_avg)
        error = np.array(test_loss_std)
        plt.errorbar(x, y=y,  color='#ff4d4d', label='Test loss')
        plt.fill_between(x, y-error, y+error,
                    alpha=0.5,  facecolor='pink', label='Test stdev')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.xlim([1, epochs])
        plt.legend()
        plt.savefig(out_path + "{}_".format(dataset_type) +
                img_name, bbox_inches='tight')


def plot_avg_cropped_px(data_file):
    x = []
    pixels_per_iteration = []
    iterations = len(data_file)
    for itr in range(iterations):
        #print("www", IMAGE_SIZE[0])
        x.append(itr+1)
        subvalues = []
        subiterations = len(data_file[itr])
        for subitr in range(subiterations):
            subvalues.append(data_file[itr][subitr]['avg_cropped_pixels'])
        pixels_per_iteration.append(stat.mean(subvalues))

    plt.clf()
    plt.margins(x=0)
    plt.xticks(x)
    plt.title("Average cropped pixels")
    rgb = (random.random(), random.random(), random.random())
    plt.bar(x, pixels_per_iteration,  align='center', color='steelblue')
    plt.ylabel("Average number of cropped pixels")
    plt.xlabel("Iterations")

    # Labels above bars
    total_pixels = pow(IMAGE_SIZE[0], 2)
    for i, v in enumerate(pixels_per_iteration):
        #plt.text(i + 0.6, v + 1, str(round(v, 2)) + "pxs / " + \
        #        str(round((v * 100) / total_pixels, 2)) + "%")
        plt.text(i + 0.6, v + 1, str(round((v * 100) / total_pixels, 2)) + "%", fontsize=6)

    plt.savefig(out_path + "avg_cropped_pixels.jpg", bbox_inches='tight')


def main():
    if len(sys.argv) <= 1:
        print("error: no input file")
        sys.exit(0)

    input_file = sys.argv[1]
    if os.path.isdir(out_path) == False:
        os.makedirs(out_path)

    data_file = []
    if os.path.isfile(input_file) == True:
        with open(input_file, 'r') as json_file:
            data_file = json.load(json_file)
            result_id = len(data_file) + 1
            json_file.close()
    else:
        print("The file {} does not exist".format(input_file))

    print("plotting...")
    dataset_type = 'test'
    plot_model_accuracy(data_file, dataset_type)
    plot_classes_results(data_file, dataset_type)
    plot_loss(data_file, dataset_type)
    plot_avg_cropped_px(data_file)

    dataset_type = 'train'
    plot_model_accuracy(data_file, dataset_type)
    plot_classes_results(data_file, dataset_type)
    plot_loss(data_file, dataset_type)

    dataset_type = 'cropped_test'
    plot_model_accuracy(data_file, dataset_type)
    plot_classes_results(data_file, dataset_type)
    plot_loss(data_file, dataset_type)


    print("Done!")

if __name__ == "__main__":
    main()
