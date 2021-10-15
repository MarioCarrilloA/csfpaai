import os.path
from torchvision import models, datasets, transforms

from nnu import *
from dsets import *


max_iterations = 25
max_subiterations = 5
epochs = 50
learning_rate = 0.1
min_accuracy = 30
num_samples = 5
dataset_name = 'CIFAR10'
 

def main():
    if len(sys.argv) <= 1:
        print("error: no input directory")
        sys.exit(0)

    input_mask_dataset = sys.argv[1]
    if os.path.isdir(input_mask_dataset) == False:
        sys.exit("ERROR: imaga dataset directory no found")

    # Init values
    out_filename = "../res/mask_results.json"
    all_metrics = []
    model = None
    metrics = None

    # Get dataset from upstream
    train_dataset, \
    test_dataset, \
    validation_dataset, \
    train_transform, \
    test_transform, \
    classes = get_dataset_components(dataset_name)

    # Test and train datasets of masks
    train_mask_dataset = croppedDataset(root=input_mask_dataset,
                                transform=transforms.ToTensor()
    )
    test_mask_dataset = croppedDataset(root=input_mask_dataset,
                                transform=transforms.ToTensor(),
                                train=False
    )

    # Loaders
    train_loader = torch.utils.data.DataLoader(
                            train_mask_dataset,
                            batch_size=128,
                            shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
                            test_mask_dataset,
                            batch_size=128,
                            shuffle=False
    )

    test_vanilla_loader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=128,
                            shuffle=False
    )

    out_trainds_img = "../res/mask_dataset_samples_train.png"
    out_testds_img = "../res/mask_dataset_samples_test.png"
    out_testvds_img = "../res/mask_dataset_samples_vanilla_test.png"
    save_dataset_samples(train_loader, out_trainds_img)
    save_dataset_samples(test_loader, out_testds_img)
    save_dataset_samples(test_vanilla_loader, out_testvds_img)


    for subitr in range(max_subiterations):
        model = None
        metrics = None
        model, metrics = build_model(
                            train_loader,
                            test_loader,
                            epochs,
                            learning_rate,
                            test_vanilla_loader,
                            classes
        )
        metrics.update({'avg_cropped_pixels': 0})
        all_metrics.append(metrics)
    collect_results(all_metrics, out_filename)

main()

