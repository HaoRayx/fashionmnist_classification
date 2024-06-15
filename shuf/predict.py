import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from shufflenet import ShuffleNetG2
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the same transform as used during training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the test dataset
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# Initialize the model and load the trained weights
model = ShuffleNetG2().to(device)
model.load_state_dict(torch.load('shufflenet_fashionmnist.pth'))
model.eval()

# Function to perform prediction and measure time
def predict_and_measure_time():
    total_correct = 0
    total_images = 0
    total_time = 0.0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

    accuracy = total_correct / total_images
    avg_time_per_image = total_time / total_images

    return accuracy, avg_time_per_image

# Perform prediction and measure time
accuracy, avg_time_per_image = predict_and_measure_time()
print('Test Accuracy: %.2f%%' % (accuracy * 100))
print('Average Prediction Time per Image: %.6f seconds' % avg_time_per_image)
