import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from shufflenet import ShuffleNetG2
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)

model = ShuffleNetG2().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_images = 0
    mini_batch_count = 0
    total_batches = len(trainloader)
    start_time = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_images += labels.size(0)
        mini_batch_count += 1
        if i % 100 == 99:
            accuracy = total_correct / total_images
            print('[Batch %d/%d, Epoch %d] loss: %.3f, Accuracy: %.2f%%' %
                  (mini_batch_count, total_batches, epoch , running_loss / 100, accuracy * 100))
            running_loss = 0.0
    end_time = time.time()
    epoch_time = end_time - start_time
    accuracy = total_correct / total_images
    print('Epoch %d, Accuracy: %.2f%%, Time: %.2fs' % (epoch , accuracy * 100, epoch_time))
    model.eval()
    total_correct_test = 0
    total_images_test = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct_test += (predicted == labels).sum().item()
            total_images_test += labels.size(0)
    test_accuracy = total_correct_test / total_images_test
    print('Test Accuracy: %.2f%%' % (test_accuracy * 100))

print('Finished Training')

torch.save(model.state_dict(), 'shufflenet_fashionmnist.pth')
