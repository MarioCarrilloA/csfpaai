from nnu import *

#**************************************************************************************************
#classes = ('plane',
#           'auto',
#           'bird',
#           'cat',
#           'deer',
#           'dog',
#           'frog',
#           'horse',
#           'ship',
#           'truck')
#
## CIFAR1-10 parameters
#input_size = 32
#num_classes = 10
#epochs=7
#out_filename = 'results.json'
#
## Check for GPU/CPU to allocate tensor
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
#    device = torch.device('cpu')
#
#
#**************************************************************************************************


MAX_ITERATIONS = 3

# Download & transform CIFAR-10 datasets
train_full_dataset = datasets.CIFAR10("./data", train=True,
                                 transform=train_transform, download=True)

test_dataset = datasets.CIFAR10("./data", train=False,
                                transform=test_transform, download=True)

# Split datasets
train_num_samples = int(len(train_full_dataset) * 0.9) # 90%(Training set).
val_num_samples = int(len(train_full_dataset) * 0.1)   #10%(Validation set)
train_dataset, validation_dataset = random_split(train_full_dataset, [train_num_samples, val_num_samples])

prev_model = None
for itr in range(MAX_ITERATIONS):
    print("Iteration: ", itr)

    # Compute base
    if itr == 0:
        print("Compute base model")
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=5000, shuffle=False)
        prev_model, pct_correct, pct_classes, trainl, testl = build_model(train_loader, test_loader, epochs)
        collect_results(epochs, pct_correct, trainl, testl, pct_classes, out_filename)
        continue

    crop_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: crop_preprocess(x, prev_model)),
    ])

    new_dataset_dir = "data_{}/".format(itr)
    csv_file = 'labels_{}.csv'.format(itr)
    print("Creating new data set...")
    print("1***train datset", len(train_dataset), type(train_dataset))
    create_new_dataset(train_dataset, new_dataset_dir, csv_file, crop_transformation)
    train_dataset = croppedCIFAR10(csv_file = csv_file, root_dir = new_dataset_dir,
                                    transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=128, shuffle=True)
    print("2***train datset", len(train_dataset), type(train_dataset))
    print("***train loader:", len(train_loader), type(train_loader))

    out_img = "dataset_samples_{}.png".format(itr)
    num_samples = 5
    save_dataset_samples(train_loader, out_img)
    prev_model, pct_correct, pct_classes, trainl, testl = build_model(train_loader, test_loader, epochs)
    save_random_samples(prev_model, num_samples, crop_transformation, train_dataset, itr)
    collect_results(epochs, pct_correct, trainl, testl, pct_classes, out_filename)

