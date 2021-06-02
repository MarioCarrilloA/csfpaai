from nnu import *

train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
         #   normalize,
         ])

# Download & transform CIFAR-10 datasets
train_full_dataset = datasets.CIFAR10("./data", train=True,
                                         transform=train_transform, download=True)

test_dataset = datasets.CIFAR10("./data", train=False,
                                        transform=test_transform, download=True)

# Split datasets
train_num_samples = int(len(train_full_dataset) * 0.9) # 90%(Training set).
val_num_samples = int(len(train_full_dataset) * 0.1)   #10%(Validation set)

print("Division: ",train_num_samples, val_num_samples)
train_dataset, validation_dataset = random_split(train_full_dataset, [train_num_samples, val_num_samples])
print("Dataset lenght: ", len(train_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=5000, shuffle=False)


model_base, pct_correct, pct_classes = build_model(train_loader, test_loader)
