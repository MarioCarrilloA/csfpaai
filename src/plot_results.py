import os.path
import json
import matplotlib.pyplot as plt
import sys
import random

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
           'truck')
out_path = "charts/"

def plot_classes_results(data_file):
    for c in range(len(classes)):
        tmp = []
        x =[]
        i = 0
        for r in data_file:
            i+=1
            tmp.append(r["classes_pcts"][c])
            x.append("i" + str(i))
        plt.clf()
        plt.title("Class accuracy: " + classes[c], fontsize=15)
        plt.ylim([0,100])
        rgb = (random.random(), random.random(), random.random())
        plt.bar(x, tmp,  align='center', color=[rgb], width=0.4)
        plt.ylabel("Percentage")
        plt.xlabel("Iterations")

        # Labels above bars
        for i, v in enumerate(tmp):
            plt.text(i - 0.20, v + 1, str(round(v, 2)) + "%")
        plt.savefig(out_path + classes[c]+".jpg", bbox_inches='tight')


def plot_model_accuracy(data_file):
    acc = []
    x =[]
    i = 0
    for r in data_file:
        i+=1
        acc.append(r["accuracy"])
        x.append("i" + str(i))
    plt.clf()
    plt.title("Model accuracy per iteration")
    plt.ylim([0,100])
    plt.bar(x, acc,  align='center', color='green', width=0.4)
    plt.ylabel("Percentage")
    plt.xlabel("Iterations")

    # Labels above bars
    for i, v in enumerate(acc):
        plt.text(i - 0.20, v + 1, str(round(v, 2)) + "%")
    plt.savefig(out_path + "model_accuracy.jpg", bbox_inches='tight')


def plot_loss(data_file):
    i = 0
    for r in data_file:
        img_name = "loss_iteration_{}.png".format(i)
        train_loss = r['train_loss']
        test_loss = r['test_loss']
        epochs = len(train_loss)
        plt.clf()
        plt.title("Train and test loss - Iteration {}".format(i))
        plt.plot(train_loss, color='blue', label='Train loss')
        plt.plot(test_loss, color='red', label='Test loss')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.xlim([0, epochs])
        plt.legend()
        plt.savefig(out_path + img_name, bbox_inches='tight')
        i+=1

def plot_avg_cropped_px(data_file):
    croppx = []
    x =[]
    i = 0
    for r in data_file:
        i+=1
        croppx.append(r['avg_cropped_pixels'])
        x.append("i" + str(i))
    plt.clf()
    plt.title("Average cropped pixels")
    #plt.ylim([0,100])
    rgb = (random.random(), random.random(), random.random())
    plt.bar(x, croppx,  align='center', color=[rgb], width=0.4)
    plt.ylabel("Average number of cropped pixels")
    plt.xlabel("Iterations")

    # Labels above bars
    for i, v in enumerate(croppx):
        plt.text(i - 0.20, v + 1, str(round(v, 2)) + "%")
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
    plot_model_accuracy(data_file)
    plot_classes_results(data_file)
    plot_loss(data_file)
    plot_avg_cropped_px(data_file)
    print("Done!")

if __name__ == "__main__":
    main()
