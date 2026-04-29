import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

######################################################################
# Three different neural networks:
# 1. FNN (only fully connected layers, 3 hidden layers)
# 2. SimpleCNN (2 convolutional layers, 2 fully connected hidden layers)
# 3. DeepCNN (3 convolutional layers with dropout, 2 fully connected hidden layers)
######################################################################

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
  
# download and load training data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# download and load test data
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
 
# create data loaders
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

######################################################################
# Network Architectures
######################################################################
 
# Network 1: Feed-Forward Network
class FNN(nn.Module):
    """
    A feed-forward neural network with only fully connected layers.

    Architecture:
        Input: 32x32x3 (3072 features)
        Hidden layer 1 (512 units, ReLU)
        Hidden layer 2 (256 units, ReLU) 
        Hidden layer 3 (128 units, ReLU)
        Output: 10 units
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
 
 
# Network 2: Simple Convolutional Neural Network
class SimpleCNN(nn.Module):
    """
    A convolutional neural network with 2 convolutional layers.

    Architecture:
        Conv layer 1 (3 input channels -> 32 filters, 3x3 kernel, padding=1)
        MaxPool 2x2 (32x32 -> 16x16)
        Conv layer 2 (32 -> 64 filters, 3x3 kernel, padding=1)
        MaxPool 2x2 (16x16 -> 8x8)
        Fully connected (64*8*8 -> 512 -> 128 -> 10)
    """
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # fully connected layers
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
 
# Network 3: Deeper CNN with Dropout
class DeepCNN(nn.Module):
    """
    A deeper CNN with 3 convolutional layers and dropout.
    Dropout randomly zeros neurons during training to prevent overfitting.

    Architecture:
        Conv layer 1 (3 -> 32 filters, 3x3, padding=1, MaxPool)
        Conv layer 2 (32 -> 64 filters, 3x3, padding=1, MaxPool)
        Conv layer 3 (64 -> 128 filters, 3x3, padding=1, MaxPool)
        Dropout (15%)
        Fully connected (128*4*4 -> 256 -> 128 -> 10)
    """
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # fully connected layers
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
 
######################################################################
# Training
######################################################################
 
def train(model, train_loader, loss_fn, optimizer, min_epochs=20, max_epochs=40):
    """
    Train a model with early stopping based on training loss.
    
    Args:
        model: The neural network to train
        train_loader: DataLoader for training data
        loss_fn: Loss function (CrossEntropyLoss for classification)
        optimizer: Updates the model weights
        min_epochs: Minimum epochs before early stopping can trigger
        max_epochs: Maximum epochs to train
    
    Returns:
        train_losses: List of training loss after each epoch
    """
    train_losses = []
    previous_loss = float('inf')
    
    for epoch in range(max_epochs):
        # set model to training mode
        model.train()

        # training
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            # calculate prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
        # calculate average loss for this epoch
        epoch_loss = calculate_training_loss(model, train_loader, loss_fn)
        train_losses.append(epoch_loss)
        
        print(f"Epoch {epoch + 1}/{max_epochs}, Training Loss: {epoch_loss:.4f}")
        
        # early stopping check only after minimum epochs
        if epoch >= min_epochs - 1:
            if epoch_loss > previous_loss:
                print(f"Early stopping as loss increased from {previous_loss:.4f} to {epoch_loss:.4f}")
                break
        
        previous_loss = epoch_loss
    
    return train_losses

def calculate_training_loss(model, train_loader, loss_fn):
    """
    Calculate total training loss with model in evaluation mode.
    """
    model.eval()
    total_loss, num_batches = 0, 0
    
    # no gradient calculation needed
    with torch.no_grad():
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # calculate loss
            total_loss += loss_fn(pred, y).item()
            num_batches += 1

    # average loss
    return total_loss / num_batches
 
######################################################################
# Testing
######################################################################
 
def test(model, test_loader):
    """
    Test model accuracy on the test set.
    
    Returns:
        accuracy: percentage of correct predictions
    """
    model.eval()
    correct = 0
    total = 0

    # no gradient calculation needed
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # get predicted class
            predicted = pred.argmax(1)
            # count correct predictions
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
 
######################################################################
# For results.pdf
######################################################################
 
def plot_loss(train_losses, title, filename):
    """
    Plot training loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss After Each Epoch: {title}')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
 
def find_correct_incorrect(model, test_loader):
    """
    Find one correctly classified and one incorrectly classified image.
    
    Returns:
        correct_example (image, true_label, predicted_label)
        incorrect_example (image, true_label, predicted_label)
    """
    model.eval()
    correct_example = None
    incorrect_example = None
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted = pred.argmax(1)
            
            for i in range(len(y)):
                true_label = y[i].item()
                pred_label = predicted[i].item()
                # found a correct prediction
                if correct_example is None and true_label == pred_label:
                    correct_example = (X[i].cpu(), true_label, pred_label)
                # found an incorrect prediction
                if incorrect_example is None and true_label != pred_label:
                    incorrect_example = (X[i].cpu(), true_label, pred_label)
                if correct_example and incorrect_example:
                    return correct_example, incorrect_example
    
    return correct_example, incorrect_example
 
def example_image(image, true_label, pred_label, filename):
    """
    Display and save one example image that was correctly classified 
    and one that was incorrectly classified
    """

    image = image.numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f'True: {classes[true_label]}, Predicted: {classes[pred_label]}')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
 
######################################################################
# Main: train and test all three networks
###################################################################### 

def main():
    # loss function: CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss()
    
    # store results
    results = {}
    
    # Train and test Network 1: FNN
    print(f"\nTraining Network 1: FNN\n")
    
    model1 = FNN().to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    
    # train the model
    train_losses1 = train(model1, train_dataloader, loss_fn, optimizer1, min_epochs=20, max_epochs=40)

    # test the model
    accuracy1 = test(model1, test_dataloader)
    print(f"\nFNN Test Accuracy: {accuracy1:.2f}%")
    
    # save results
    results['FNN'] = {'accuracy': accuracy1, 'train_losses': train_losses1}
    
    # plot and save training loss
    plot_loss(train_losses1, 'FNN', 'fnn_loss.png')
    
    # find and save example images
    correct1, incorrect1 = find_correct_incorrect(model1, test_dataloader)
    if correct1:
        example_image(*correct1, 'fnn_correct.png')
    if incorrect1:
        example_image(*incorrect1, 'fnn_incorrect.png')
    
    torch.save(model1.state_dict(), 'fnn_model.pth')
    
    # Train and test Network 2: SimpleCNN
    print(f"\nTraining Network 2: SimpleCNN\n")
    
    model2 = SimpleCNN().to(device)

    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
    
    train_losses2 = train(model2, train_dataloader, loss_fn, optimizer2, min_epochs=25, max_epochs=50)
    
    accuracy2 = test(model2, test_dataloader)
    print(f"\nSimpleCNN Test Accuracy: {accuracy2:.2f}%")
    
    results['SimpleCNN'] = {'accuracy': accuracy2, 'train_losses': train_losses2}
    
    plot_loss(train_losses2, 'SimpleCNN', 'simplecnn_loss.png')
    
    correct2, incorrect2 = find_correct_incorrect(model2, test_dataloader)
    if correct2:
        example_image(*correct2, 'simplecnn_correct.png')
    if incorrect2:
        example_image(*incorrect2, 'simplecnn_incorrect.png')
    
    torch.save(model2.state_dict(), 'simplecnn_model.pth')
    
    # Train and test Network 3: DeepCNN
    print(f"\nTraining Network 3: DeepCNN\n")
    
    model3 = DeepCNN().to(device)
 
    optimizer3 = optim.SGD(model3.parameters(), lr=0.01)
    
    train_losses3 = train(model3, train_dataloader, loss_fn, optimizer3, min_epochs=40, max_epochs=60)
    
    accuracy3 = test(model3, test_dataloader)
    print(f"\nDeepCNN Test Accuracy: {accuracy3:.2f}%")
    
    results['DeepCNN'] = {'accuracy': accuracy3, 'train_losses': train_losses3}
    
    plot_loss(train_losses3, 'DeepCNN', 'deepcnn_loss.png')
    
    correct3, incorrect3 = find_correct_incorrect(model3, test_dataloader)
    if correct3:
        example_image(*correct3, 'deepcnn_correct.png')
    if incorrect3:
        example_image(*incorrect3, 'deepcnn_incorrect.png')
    
    torch.save(model3.state_dict(), 'deepcnn_model.pth')
    
    # print Summary
    print(f"\nFinal Results:")
    print(f"\n{'Network':<20} {'Accuracy':<15} {'Epochs Trained':<15}\n")
    for name, data in results.items():
        print(f"{name:<20} {data['accuracy']:.2f}%{'':<8} {len(data['train_losses']):<15}")

if __name__ == "__main__":
    main()