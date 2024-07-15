import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import io
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os

current_dir =  os.getcwd()
test_file = 'test-00000-of-00001-44110b9df98c5585.parquet'
data_path = os.path.join(current_dir,'Data',test_file)

# Load the Parquet file
test_df = pd.read_parquet(data_path, engine='pyarrow')
test_df.head()

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
test_df['image_np'] = test_df['image'].apply(decode_image)

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

test_dataset = CustomDataset(test_df, transform=transform) 
batch_size = 32
# DataLoader for testing set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load the VGG-19 model with pre-trained weights
model = vgg19(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 4)

# Path to the saved model
model_path = 'vgg19_trained_model.pth'
path = os.path.join(os.getcwd(), model_path)

# Load Model
try:
    #device to load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)  # Move the model to the device
    model.eval()
    print(f"Model loaded successfully on {device}")
except RuntimeError as e:
    print(f"Failed to load the model: {e}")

#lists to store predictions and ground truths
all_predictions = []
all_labels = []

#Evaluate on test set
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs_rgb = torch.cat([inputs, inputs, inputs], dim=1) 
        inputs_rgb = inputs_rgb.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs_rgb)
        _, predicted = torch.max(outputs.data, 1)

        # Append predictions and labels
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = np.mean(all_predictions == all_labels) * 100
print(f"Accuracy on Test Set: {accuracy:.2f}%")

# Create confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()