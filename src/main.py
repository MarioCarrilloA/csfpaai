from nnu import *

exp = Experiment('PAAL Experiment')
EXP_FOLDER = '../exp/'
log_location = os.path.join(EXP_FOLDER, os.path.basename(sys.argv[0])[:-3])
if len(exp.observers) == 0:
    print('Adding a file observer in %s' % log_location)
    exp.observers.append(file_storage.FileStorageObserver.create(log_location))


@exp.config
def config():
    max_iterations = 20
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

    # Download & transform CIFAR-10 datasets
    train_full_dataset = datasets.CIFAR10(
                    "./data",
                    train=True,
                    transform=train_transform,
                    download=True
    )
    test_dataset = datasets.CIFAR10(
                    "./data",
                    train=False,
                    transform=test_transform,
                    download=True
    )

    # Split datasets in 90% for training set and 10% for Validation set.
    train_num_samples = int(len(train_full_dataset) * 0.9)
    val_num_samples = int(len(train_full_dataset) * 0.1)
    train_dataset, validation_dataset = random_split(
            train_full_dataset,
            [train_num_samples, val_num_samples]
    )
    prev_model = None
    out_filename = "../res/results.json"
    for itr in range(max_iterations):
        print("Iteration: ", itr)
        cropped_pixels = []
        new_dataset_dir = "data_{}/".format(itr)
        csv_file = 'labels_{}.csv'.format(itr)
        out_img = "../res/dataset_samples_{}.png".format(itr)

        # Compute base
        if itr == 0:
            print("Compute base model")
            train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=128,
                            shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=5000,
                            shuffle=False
            )
            # Train model
            prev_model, metrics = build_model(
                                train_loader,
                                test_loader,
                                epochs,
                                learning_rate
            )
            #_run.log_scalar(1, metrics)
            metrics.update({'avg_cropped_pixels': 0})
            collect_results(metrics, out_filename)
            save_dataset_samples(train_loader, out_img)
            continue

        # When the base model has been trained.
        ######################################################################
        crop_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: crop_preprocess(x, prev_model, cropped_pixels))])

        save_random_samples(
                prev_model,
                num_samples,
                crop_transformation,
                train_dataset,
                itr
        )
        print("Creating new data set...")
        # This line is to avoid saving the images with the transformations
        # used for data augmentation.
        if isinstance(train_dataset, croppedCIFAR10):
            train_dataset.transform = None
        else:
            train_dataset.dataset.transform = None


        # Prepare dataset and loaders for new iteration
        create_new_dataset(
                train_dataset,
                new_dataset_dir,
                csv_file,
                crop_transformation
        )
        train_dataset = croppedCIFAR10(
                        root=new_dataset_dir,
                        transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=128,
                        shuffle=True
        )
        prev_model, metrics = build_model(
                            train_loader,
                            test_loader,
                            epochs,
                            learning_rate)
        #_run.log_scalar(1, metrics)
        avg_cpix = sum(cropped_pixels) / len(cropped_pixels)
        metrics.update({'avg_cropped_pixels': avg_cpix})
        collect_results(metrics, out_filename)
        save_dataset_samples(train_loader, out_img)

        if all(vp <= min_accuracy for vp in metrics['testds_classes_pcts']):
            break
