from nnu import *

# CIFAR1-10 parameters
input_size = 32
num_classes = 10
epochs=50
out_filename = 'results.json'

# Check for GPU/CPU to allocate tensor
device = 'cuda'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
])

test_transform = transforms.Compose([
        transforms.ToTensor(),
])

epochs=50
out_filename = 'results.json'

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


# Train model and collect data
model_base, pct_correct, pct_classes, trainl, testl = build_model(train_loader, test_loader, epochs)
collect_results(epochs, pct_correct, trainl, testl, pct_classes, out_filename)

#########################################################################################################
# SECOND ATTEMP
#########################################################################################################

print("------------------------------------------------")
print("Executing new iteration")
crop_transformation_a1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),#########################
    transforms.RandomHorizontalFlip(),##################################
    #transforms.Lambda(lambda x: crop_preprocess(x, model_base)),
    transforms.ToTensor(),##############################
    transforms.Lambda(lambda x: crop_preprocess(x, model_base)),####################
])

train_full_dataset_a1 = datasets.CIFAR10("./data", train=True,
                                 transform=crop_transformation_a1, download=True)

test_dataset_a1 = datasets.CIFAR10("./data", train=False,
                                transform=test_transform, download=True)

# Split datasets
train_num_samples_a1 = int(len(train_full_dataset_a1) * 0.9) # 90%(Training set).
val_num_samples_a1 = int(len(train_full_dataset_a1) * 0.1)   #10%(Validation set)
train_dataset_a1, validation_dataset_a1 = random_split(train_full_dataset_a1, [train_num_samples_a1, val_num_samples_a1])

# Loaders
train_loader_a1 = torch.utils.data.DataLoader(train_full_dataset_a1,
                                           batch_size=128, shuffle=True)
test_loader_a1 = torch.utils.data.DataLoader(test_dataset_a1,
                                          batch_size=5000, shuffle=False)

model_a1, pct_correct_a1, pct_classes_a1, trainl, testl = build_model(train_loader_a1, test_loader_a1, epochs)
collect_results(epochs, pct_correct_a1, trainl, testl, pct_classes_a1, out_filename)
