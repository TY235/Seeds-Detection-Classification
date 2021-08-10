import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from sklearn.model_selection import KFold
import copy
import random
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 3)
        # self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(6, 12, 5, 3) # 64, 10, 5)
        self.conv3 = nn.Conv2d(12, 24, 5, 3) # 64, 10, 5)
        #self.conv3 = nn.Conv2d(64, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = nn.Linear(3136, 128) # 222784  # 3600   # [20 x 3136], m2: [222784 x 128]
        self.fc1 = nn.Linear(1536, 128) # 1600, 128) # 222784  # 3600   # [20 x 3136], m2: [222784 x 128]
##        self.fc1 = nn.Linear(7776, 128) #512^2 image
        self.fc2 = nn.Linear(128, 1)
        # m1: [20 x 64], m2: [3136 x 128]
        # m1: [20 x 1600], m2: [64 x 128]

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)  # torch.tanh(x) # F.relu(x)
        # x = F.max_pool2d(x, 4)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        #x = self.conv3(x)
        #x = torch.tanh(x) # x = F.relu(x)
        #x = F.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
##        print(x.shape)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Arguments:
    k_folds = 5
    num_epochs = 40
    sched_step_size = 5
    gamma = 0.9
    lr = 1e-3

def train(args, model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        #print(batch_idx)
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output.view(-1), target.float())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / (batch_idx + 1)

def main():
    args = Arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomVerticalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.ImageFolder(root="dataset/train", transform=train_transforms)
    dataset_validation = datasets.ImageFolder(root="dataset/validation", transform=test_transforms)
    dataset_test = datasets.ImageFolder(root="dataset/test", transform=test_transforms)

    best_model = None
    best_total_val = 0
    best_correct_val = 0
    best_model_acc = 0

##    for f, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
##        print(f'Fold number {f}')
##        train_subsampler = SubsetRandomSampler(train_ids)
##        val_subsampler = SubsetRandomSampler(val_ids)



    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        sampler=RandomSampler(dataset_train),
        num_workers = 0
##        worker_init_fn=seed_worker
##            sampler=train_subsampler
    )

    val_loader = DataLoader(
        dataset_validation,
        batch_size=64,
        #sampler=RandomSampler(dataset_validation),
        num_workers = 0
##        worker_init_fn=seed_worker
##            sampler=val_subsampler
    )

    net = Net()
    net = net.to(device)

    # learning rate reccomended by andrej kaparthy
    # http://karpathy.github.io/2019/04/25/recipe/
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    scheduler = StepLR(optimizer, step_size=args.sched_step_size, gamma=args.gamma)

    num_epoch = 0
    num_epoch_not_improved = 0
    while num_epoch_not_improved < 20:
        num_epoch += 1

        #train
        print(f'Epoch number {num_epoch}')
        train_loss = train(args, net, device, train_loader, optimizer, criterion)
        print(f'Loss: {train_loss}')
        scheduler.step()

        #evaluate
        net.eval()


        correct_val, total_val = 0, 0
        print('Evaluate on validation set')
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(val_loader):
                input = input.to(device)
                target = target.to(device).view(-1)

                out = net(input)
                predicted = (out.data > 0.5).float().view(-1)
                total_val += predicted.size(0)
                #print(predicted == target)
                correct_val += (predicted == target).sum().item()
                #scheduler.step()
        acc = correct_val / total_val
        print('\t{0:.2f}% ({1}/{2})'.format(acc*100, correct_val, total_val))

        if acc > best_model_acc:
            best_model = copy.deepcopy(net)
            best_model_acc = acc
            best_total_val = total_val
            best_correct_val = correct_val
            num_epoch_not_improved = 0
        else:
            num_epoch_not_improved += 1

        print('Current best model validation:')
        print('\t{0:.2f}% ({1}/{2})'.format(best_model_acc*100, best_correct_val, best_total_val))
        print()

    test_loader = DataLoader(
        dataset_test,
        batch_size=64,
    )

    best_model.eval()

    correct_train, total_train = 0, 0
    print('Evaluate on training set')
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device).view(-1)

            out = best_model(input)
            predicted = (out.data > 0.5).float().view(-1)
            total_train += predicted.size(0)
            #print(predicted == target)
            correct_train += (predicted == target).sum().item()
            #scheduler.step()
    acc = correct_train / total_train
    print('\t{0:.2f}% ({1}/{2})'.format(acc*100, correct_train, total_train))

    correct_test, total_test = 0, 0
    print('Evaluate on test set')
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device).view(-1)

            out = best_model(input)
            predicted = (out.data > 0.5).float().view(-1)
            total_test += predicted.size(0)
            #print(predicted == target)
            correct_test += (predicted == target).sum().item()
            scheduler.step()
    test_acc = correct_test / total_test
    print('\t{0:.2f}% ({1}/{2})'.format(test_acc*100, correct_test, total_test))
    torch.save(best_model.state_dict(), 'out.pth')

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    main()
