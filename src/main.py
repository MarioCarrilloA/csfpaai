import os.path
from torchvision import models, datasets, transforms

from nnu import *
from dsets import *

exp = Experiment('PAAL Experiment')
EXP_FOLDER = '../exp/'
log_location = os.path.join(EXP_FOLDER, os.path.basename(sys.argv[0])[:-3])

if len(exp.observers) == 0:
    print('Adding a file observer in %s' % log_location)
    exp.observers.append(file_storage.FileStorageObserver.create(log_location))


@exp.config
def config():
    max_iterations = 25
    epochs = 50
    learning_rate = 0.1
    min_accuracy = 30
    num_samples = 5


@exp.automain
def main(
    max_iterations,
    epochs,
    learning_rate,
    min_accuracy,
    num_samples
):

    train_dataset, test_dataset, validation_dataset, train_transform, test_transform, classes = get_dataset_components('CIFAR10')
    test_transformed_dataset = test_dataset
    prev_model = None
    out_filename = "../res/results.json"
    for itr in range(max_iterations):
        print("Iteration: ", itr)
        cropped_pixels = []
        new_dataset_dir = "data_{}/".format(itr)
        out_img = "../res/dataset_samples_{}.png".format(itr)
        ext_out_img = "../res/ctestds_dataset_samples_{}.png".format(itr)

        # Compute base when our iteration is 0
        if itr == 0:
            print("Compute base model")
            train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=128,
                            shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=128,
                            shuffle=False
            )

            test_transformed_loader = test_loader
            # Train model
            prev_model, metrics = build_model(
                                train_loader,
                                test_loader,
                                epochs,
                                learning_rate,
                                test_transformed_loader,
                                classes
            )
            #_run.log_scalar(1, metrics)
            metrics.update({'avg_cropped_pixels': 0})
            collect_results(metrics, out_filename)
            save_dataset_samples(train_loader, out_img)
            save_dataset_samples(test_transformed_loader, ext_out_img)
            continue


        # When the base model has been trained and our iteration is
        # greater than 0 we are going to work with cropped data.
        else:
            # Set algorithm to compute  class-specific activation of
            # convolutional layers.
            extractor = GradCAM(prev_model, 'resnet.layer4', 'resnet.fc')

            # Use the class activation map algorithm and the previous
            # trained model to specify the transformation for next iteration.
            crop_transformation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: crop_preprocess(
                                            x,
                                            prev_model,
                                            extractor,
                                            cropped_pixels))
            ])

            # Save some samples using the our customized tranformation
            save_random_samples(
                    prev_model,
                    extractor,
                    num_samples,
                    crop_transformation,
                    train_dataset,
                    itr,
                    test_transform,
                    classes
            )

            save_sequential_samples(
                    prev_model,
                    extractor,
                    num_samples,
                    crop_transformation,
                    test_transformed_loader,
                    itr,
                    classes
            )


            # This line is to avoid saving the images with the transformations
            # used for data augmentation.
            if isinstance(train_dataset, croppedDataset):
                train_dataset.transform = None
            else:
                train_dataset.dataset.transform = None

            # Prepare dataset and loaders for new iteration
            print("Create new TRAIN dataset...")
            create_new_dataset(
                    train_dataset,
                    new_dataset_dir,
                    crop_transformation
            )
            print("Creating new TEST dataset")
            test_transformed_dataset.transform = None
            create_new_dataset(
                    test_transformed_dataset,
                    new_dataset_dir,
                    crop_transformation,
                    train=False
            )

            # Remove extra transformations and only allow tensor conversion
            test_transformed_dataset.transform = transforms.ToTensor()

            # Create datasets with the cropped data obtained from
            # previous iteration.
            train_dataset = None
            train_dataset = croppedDataset(
                            root=new_dataset_dir,
                            transform=train_transform
            )
            train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=128,
                            shuffle=True
            )

            test_transformed_dataset = None
            test_transformed_dataset = croppedDataset(
                            root=new_dataset_dir,
                            transform=test_transform,
                            train=False
            )
            test_transformed_loader = torch.utils.data.DataLoader(
                            test_transformed_dataset,
                            batch_size=128,
                            shuffle=False
            )

            prev_model = None
            prev_model, metrics = build_model(
                                train_loader,
                                test_loader,
                                epochs,
                                learning_rate,
                                test_transformed_loader,
                                classes
            )

            #_run.log_scalar(1, metrics)
            avg_cpix = sum(cropped_pixels) / len(cropped_pixels)
            metrics.update({'avg_cropped_pixels': avg_cpix})
            collect_results(metrics, out_filename)
            save_dataset_samples(train_loader, out_img)
            save_dataset_samples(test_transformed_loader, ext_out_img)

            if all(vp <= min_accuracy for vp in metrics['testds_classes_pcts']):
                break
