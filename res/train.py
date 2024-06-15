import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
import resmodel
import utils
import time
import argparse
import os
import csv
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='resnet50', help="model")
parser.add_argument("--patience", type=int, default=7, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--nepochs", type=int, default=10, help="max epochs")
parser.add_argument("--nworkers", type=int, default=1, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--data", type=str, default='FashionMNIST', help="MNIST, or FashionMNIST")
args = parser.parse_args()

#viz
# tsboard = SummaryWriter()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
    run += 1
    current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)



# Define transforms.
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create dataloaders. Use pin memory if cuda.

if args.data == 'FashionMNIST':
    trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.FashionMNIST('./data', train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on FashionMNIST')
else:
    trainset = datasets.MNIST('./data-mnist', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.MNIST('./data-mnist', train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on MNIST')


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)

        # Zero the gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)



if __name__ == '__main__':

    # Init network, criterion and early stopping
    model = resmodel.__dict__[args.model]().to(device)
    criterion = torch.nn.CrossEntropyLoss()


    # Define optimizer
    optimizer = optim.Adam(model.parameters())

    # Train the network
    patience = args.patience
    best_loss = 1e4
    writeFile = open('{}/stats.csv'.format(current_dir), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model(model, train_loader,
                                      criterion, optimizer)

        val_start = time.time()
        val_loss, val_acc = run_model(model, val_loader,
                                      criterion, optimizer, False)

        end = time.time()

        # print stats
        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s\t average val time {:.8f}s""".format(e+1, train_loss, train_acc, val_loss,
                                        val_acc, end - start,(end-val_start)/len(valset))
        print(stats)

        # viz
        # tsboard.add_scalar('data/train-loss',train_loss,e)
        # tsboard.add_scalar('data/val-loss',val_loss,e)
        # tsboard.add_scalar('data/val-accuracy',val_acc.item(),e)
        # tsboard.add_scalar('data/train-accuracy',train_acc.item(),e)


        # Write to csv file
        writer.writerow([e+1, train_loss, train_acc.item(), val_loss, val_acc.item()])
        # early stopping and save best model
    torch.save(model.state_dict(), 'res50.pth')
