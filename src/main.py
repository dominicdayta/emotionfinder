import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.datasets import FER2013
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

import mlflow
import mlflow.pytorch
import os
import random
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# mlflow ui --host 0.0.0.0 --port 5000
mlflow.set_tracking_uri("http://localhost:5000") # Adjust if your server is on a different IP/port

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 1 input channel (grayscale)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 48x48 -> 24x24

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 24x24 -> 12x12

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 12x12 -> 6x6

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Added layer
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 6x6 -> 3x3

        self.flatten = nn.Flatten()
        # The input features to the fc layer depend on the output size of the last pooling layer
        # For 3x3 output from pool4 with 256 channels: 256 * 3 * 3
        self.fc1 = nn.Linear(256 * (img_size // 16) * (img_size // 16), 1024) # //16 because 4 pooling layers (2*2*2*2)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, img_size, subset=None, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        nrow, ncol = self.data_frame.shape
        self.labels = self.data_frame[self.data_frame.columns[(ncol - 7):(ncol + 1)]]
        self.data_frame.drop(self.data_frame.columns[(ncol - 7):(ncol + 1)], axis=1, inplace=True)

        if(subset is not None):
            self.data_frame = self.data_frame.iloc[subset]
            self.labels = self.labels.iloc[subset]
        self.transform = transform
        
        # Convert pixel strings to numpy arrays
        self.pixels = self.data_frame.to_numpy(dtype=np.uint8).reshape(-1, img_size, img_size)
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        image = self.pixels[idx,:,:]
        label = self.labels.iloc[idx,:].to_numpy().argmax()

        image = Image.fromarray(image).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 3. Training Function for a Single MLflow Run ---
def train_model(run_name, params):
    # Set a unique random seed for each run for clearer distinction
    set_seed(params['random_seed'])

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)

        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device.type)
        print(f"[{run_name}] Using device: {device}")

        # Model, Loss, Optimizer
        model = SimpleCNN(num_classes=params['num_classes'], img_size=params['img_size']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5) # Added weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # Learning rate scheduler

        # Data
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizing for grayscale
        ])

        # For validation and test data, only basic transforms
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = FER2013Dataset(csv_file=params['train_csv_path'], img_size=params['img_size'], subset=list(range(0,20001,1)), transform=train_transform)
        val_dataset = FER2013Dataset(csv_file=params['train_csv_path'], img_size=params['img_size'], subset=list(range(28001,28709,1)), transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

        for epoch in range(params['epochs']):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = 100 * correct_train / total_train
            mlflow.log_metric(f"loss", epoch_train_loss, step=epoch)
            mlflow.log_metric(f"acc", epoch_train_acc, step=epoch)

            # --- Validation Phase ---
            model.eval() # Set model to evaluation mode
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_val_labels = []
            all_val_preds = []

            with torch.no_grad(): # No gradients needed for validation
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(predicted.cpu().numpy())


            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_acc = 100 * correct_val / total_val
            mlflow.log_metric(f"val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric(f"val_acc", epoch_val_acc, step=epoch)

            # Learning rate scheduling
            scheduler.step(epoch_val_loss)

        # Log final metrics
        mlflow.log_metric("final_loss", epoch_train_loss)
        mlflow.log_metric("final_acc", epoch_train_acc)
        mlflow.log_metric("final_val_loss", epoch_val_loss)
        mlflow.log_metric("final_val_acc", epoch_val_acc)

        # Log the model
        mlflow.pytorch.log_model(model, "model", registered_model_name=f"SimpleCNN_{run_name}")
        print(f"[{run_name}] Model logged and registered.")
        print(f"[{run_name}] MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"[{run_name}] MLflow UI Link: {mlflow.active_run().info.artifact_uri}")

# --- 4. Main Execution for Parallel Runs ---
if __name__ == "__main__":
    print("Starting parallel MLflow runs...")

    # Define parameters for each run
    run_configs = [
        {
            "run_name": "Run_A",
            "params": {
                "train_csv_path": "../data/fer2013/fer2013_training_onehot.csv",
                "num_classes": 7,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 500,
                "img_size": 48,
                "random_seed": 42
            }
        },
        {
            "run_name": "Run_B",
            "params": {
                "train_csv_path": "../data/fer2013/fer2013_training_onehot.csv",
                "num_classes": 7,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 500,
                "img_size": 48,
                "random_seed": 123
            }
        },
        {
            "run_name": "Run_C",
            "params": {
                "train_csv_path": "../data/fer2013/fer2013_training_onehot.csv",
                "num_classes": 7,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 500,
                "img_size": 48,
                "random_seed": 28
            }
        },
        {
            "run_name": "Run_D",
            "params": {
                "train_csv_path": "../data/fer2013/fer2013_training_onehot.csv",
                "num_classes": 7,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 500,
                "img_size": 48,
                "random_seed": 28
            }
        }
    ]

    # Use ThreadPoolExecutor to run tasks in parallel
    # Max workers determines how many runs can happen concurrently.
    # Be mindful of your CPU/GPU resources.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for config in run_configs:
            # Each submit call starts a new thread, running train_model
            future = executor.submit(train_model, config['run_name'], config['params'])
            futures.append(future)

        # Wait for all runs to complete
        for future in futures:
            future.result() # This will re-raise any exceptions that occurred in the thread

    print("\nAll parallel MLflow runs completed.")
