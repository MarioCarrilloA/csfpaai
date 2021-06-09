import os.path
import json
import matplotlib.pyplot as plt

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

def plot_CIFAR10_results():
    #TODO: Optimze this code
    out_path = "charts/"
    if os.path.isdir(out_path) == False:
        os.makedirs(out_path)

    file_name = 'results.json'
    data_file = []
    if os.path.isfile(file_name) == True:
        with open(file_name, 'r') as json_file:
            data_file = json.load(json_file)
            result_id = len(data_file) + 1
            json_file.close()
    else:
        print("The file {} does not exist".format(file_name))

    # Plot classes comparison
    for c in range(len(classes)):
        tmp = []
        x =[]
        i = 0
        for r in data_file:
            i+=1
            tmp.append(r["classes_pcts"][c])
            x.append("i" + str(i))
        plt.clf()
        plt.title("Class accuracy: " + classes[c])
        plt.ylim([0,100])
        plt.bar(x, tmp,  align='center', color='blue', width=0.4)
        plt.ylabel("Percentage")
        plt.xlabel("Iterations")
        plt.savefig(out_path + classes[c]+".jpg", bbox_inches='tight')
    ###################################################################
    # Plot General accuracy
    ##################################################################
    acc = []
    for r in data_file:
        acc = r["accuracy"]
    plt.clf()
    plt.title("Model accuracy per iteration")
    plt.ylim([0,100])
    plt.bar(x, acc,  align='center', color='green', width=0.4)
    plt.ylabel("Percentage")
    plt.xlabel("Iterations")
    plt.savefig(out_path + "model_accuracy.jpg", bbox_inches='tight')


plot_CIFAR10_results()
