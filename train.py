import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,SubsetRandomSampler
import io
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os

current_dir =  os.getcwd()
train_file = 'train-00000-of-00001-c08a401c53fe5312.parquet'
data_path = os.path.join(current_dir,'Data',train_file)

# Load the Parquet file
df = pd.read_parquet(data_path, engine='pyarrow')
df.head()

# Function to decode image bytes and convert to NumPy array
def decode_image(image_dict):
    try:
        byte_string = image_dict['bytes']
        image_stream = io.BytesIO(byte_string)
        image_pil = Image.open(image_stream)
        image_np = np.array(image_pil)
        return image_np
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
df['image_np'] = df['image'].apply(decode_image)

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx]["image_np"]
        label = self.df.iloc[idx]["label"]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
dataset = CustomDataset(df, transform=transform)

#Split data 
dataset_size = len(df)
indices = list(range(dataset_size))
split = int(0.9 * dataset_size)

train_sampler = SubsetRandomSampler(indices[:split])
valid_sampler = SubsetRandomSampler(indices[split:])
# Create data loaders
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=2)
valid_loader = DataLoader(dataset, batch_size=32, sampler=valid_sampler, num_workers=2)

#VGG-19 model with pre-trained weights
model = vgg19(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

num_epochs = 10  #epochs

for epoch in range(num_epochs):
    model.train()  # training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Training loop
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs_rgb = torch.cat([inputs, inputs, inputs], dim=1) 
        inputs_rgb, labels = inputs_rgb.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs_rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Store training loss and accuracy for the epoch
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_accuracy = 100 * correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    #epoch summary
    print(f"Epoch {epoch + 1} completed. Total Train Loss: {epoch_train_loss:.3f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

    # Validation after each epoch
    model.eval()  #evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs_rgb = torch.cat([inputs, inputs, inputs], dim=1)
            inputs_rgb, labels = inputs_rgb.to(device), labels.to(device)

            outputs = model(inputs_rgb)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(valid_loader)
    epoch_val_accuracy = 100 * val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    # validation accuracy
    print(f"Validation Epoch {epoch + 1} Loss: {epoch_val_loss:.3f}, Validation Accuracy: {epoch_val_accuracy:.2f}%")

# final training accuracy
final_train_accuracy = train_accuracies[-1]
print(f"Training finished. Final Train Accuracy: {final_train_accuracy:.2f}%")

# Plotting the training and validation metrics
epochs = range(1, num_epochs + 1)

# Plotting Losses
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, '-o', label='Training Loss')
plt.plot(epochs, val_losses, '-o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.xticks(epochs)
plt.tight_layout()
plt.show()

# Plotting Accuracies
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, '-o', label='Training Accuracy')
plt.plot(epochs, val_accuracies, '-o', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.xticks(epochs)
plt.tight_layout()
plt.show()