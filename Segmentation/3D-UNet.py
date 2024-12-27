# %%

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# %%

def load_nifti_file(filepath):
    """Load a NIfTI file (.nii.gz) and return the data as a float32 array."""
    try:
        data = nib.load(filepath).get_fdata().astype(np.float32)
        return data
    except FileNotFoundError:
        # print(f"File not found: {filepath}")
        return None
    except Exception as e:
        # print(f"Error loading file {filepath}: {str(e)}")
        return None


# %%

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_volume(volume):
    """Normalize a 3D volume to the range [0, 1]."""
    assert isinstance(volume, np.ndarray), "Volume must be a numpy array."
    data_scaler = MinMaxScaler()  # Explicitly use MinMaxScaler for data
    volume_flat = volume.flatten().reshape(-1, 1)
    volume_scaled_flat = data_scaler.fit_transform(volume_flat)
    return volume_scaled_flat.reshape(volume.shape)

# %%

def gamma_correction(volume, gamma_values):
    """Apply gamma correction to specific modalities."""
    assert isinstance(volume, np.ndarray), "Volume must be a numpy array."
    corrected_volume = np.empty_like(volume)
    for i, gamma in enumerate(gamma_values):
        corrected_volume[..., i] = np.power(volume[..., i], gamma)
    return corrected_volume

# %%

def window_setting_operation(volume, window_width, window_level):
    """Apply Window Setting Operation (WSO) to the volume."""
    assert isinstance(volume, np.ndarray), "Volume must be a numpy array."
    U = np.max(volume)
    W = window_width / U
    b = U * (window_level / window_width - 0.5)

    # Apply WSO
    return np.maximum(W * volume + b, 0)


# %%

def crop_volume(volume, crop_size=(128, 128, 155)):
    """Crop the volume to a specific size."""
    assert isinstance(volume, np.ndarray), "Volume must be a numpy array."
    assert len(volume.shape) >= 3, "Volume must have at least 3 dimensions."

    # Check original volume shape
    original_shape = volume.shape

    # Calculate cropping indices
    center = np.array(original_shape[:3]) // 2
    start = center - np.array(crop_size) // 2
    end = start + np.array(crop_size)

    # Ensure we don't go out of bounds
    start = np.clip(start, 0, None)
    end = np.clip(end, None, original_shape[:3])

    # Crop and return the volume
    return volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


# %%

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, crop_size=(128, 128, 128), transform=None, is_training=True):
        self.data_dir = data_dir
        self.crop_size = crop_size  # Updated crop size to reflect new depth
        self.transform = transform
        self.is_training = is_training
        self.cases = os.listdir(data_dir)

    def __len__(self):
        return len(self.cases)

    def preprocess_case(self, case_path):
        case = os.path.basename(case_path)

        # Construct paths for modalities
        t1ce_path = os.path.join(case_path, f'{case}_t1ce.nii')
        t2_path = os.path.join(case_path, f'{case}_t2.nii')
        flair_path = os.path.join(case_path, f'{case}_flair.nii')
        mask_path = os.path.join(case_path, f'{case}_seg.nii') if self.is_training else None

        # Load modalities and check if each file exists
        try:
            t1ce = load_nifti_file(t1ce_path)
            t2 = load_nifti_file(t2_path)
            flair = load_nifti_file(flair_path)
            mask = load_nifti_file(mask_path) if self.is_training else None
        except FileNotFoundError as e:
            return None, None

        # Ensure all required modalities are loaded
        modalities = [t1ce, t2, flair]
        loaded_modalities = [normalize_volume(mod) for mod in modalities if mod is not None]
        if len(loaded_modalities) < 3:
            return None, None

        # Apply gamma correction and window setting operation
        gamma_values = [2.9, 3.2, 1.0]  # T1CE, T2, FLAIR (FLAIR unchanged)
        corrected_volumes = gamma_correction(np.stack(loaded_modalities, axis=-1), gamma_values)
        corrected_volumes = window_setting_operation(corrected_volumes, 255, 128)

        # Crop to the required depth (remove first 15 and last 12 slices)
        corrected_volumes = corrected_volumes[:, :, 15:143]

        # Crop the volume spatially
        cropped_volume = crop_volume(corrected_volumes, self.crop_size)

        if self.is_training:
            if mask is not None:
                mask[mask == 4] = 3   # Reassign label 4 to 3 in the mask.
                mask = mask[:, :, 15:143]  # Crop the mask to match the depth
                cropped_mask = crop_volume(mask, self.crop_size)
                cropped_mask = torch.tensor(cropped_mask, dtype=torch.long).permute(2, 0, 1)
            else:
                return None, None  # Skip this case if mask is missing
        else:
            cropped_mask = None

        return cropped_volume, cropped_mask

    def __getitem__(self, idx):
        case_path = os.path.join(self.data_dir, self.cases[idx])
        image, mask = self.preprocess_case(case_path)

        # If a case is skipped (missing files), try the next item
        if image is None or (self.is_training and mask is None):
            return self.__getitem__((idx + 1) % len(self.cases))

        # Convert to torch tensors and permute to (channels, depth, height, width)
        image = torch.tensor(image, dtype=torch.float32).permute(3, 2, 0, 1)  # Shape (Height, Width, Depth, Channels) -> (Channels, Depth, Height, Width)

        # Apply transformations if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        return (image, mask) if self.is_training else image


# %%

# Define paths and dataset
data_dir = '/mnt/SSD2/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
crop_size = (128, 128, 128)
dataset = BrainTumorDataset(data_dir, crop_size=crop_size)

# Split into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
batch_size = 24  # Adjust based on available memory
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



# %%


# Check the shape and dtype of one batch from each DataLoader
def check_batch_shapes_and_dtypes(loader, name):
     for images, masks in loader:
         print(f"{name} - Images shape: {images.shape}, Images dtype: {images.dtype}")
         print(f"{name} - Masks shape: {masks.shape}, Masks dtype: {masks.dtype}\n")
         break  # Only check the first batch to inspect dimensions and dtype

# Checking train, validation, and test set shapes and dtypes
check_batch_shapes_and_dtypes(train_loader, "Train")
check_batch_shapes_and_dtypes(val_loader, "Validation")
check_batch_shapes_and_dtypes(test_loader, "Test")



# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.colors as mcolors

# Custom colormap for segmentation mask
tumor_colors = {
    0: (0.0, 0.0, 0.0),  # Background (black)
    1: (1.0, 0.0, 0.0),  # Necrotic/non-enhancing tumor core (red)
    2: (0.0, 1.0, 0.0),  # Peritumoral edema (green)
    3: (0.0, 0.0, 1.0),  # GD-enhancing tumor (blue)
}
cmap = mcolors.ListedColormap([tumor_colors[i] for i in tumor_colors.keys()])

def visualize_samples_with_colors(train_loader, val_loader):
    # Visualize 5 training samples
    print("Visualizing 5 Training Samples with Tumor Masks:")
    train_count = 0
    for images, masks in train_loader:
        for i in range(min(5, images.shape[0])):  # Ensure we only take up to 5 samples
            image = images[i].cpu().numpy()
            mask = masks[i].cpu().numpy()

            # Check tensor shape and extract the central slice for each modality
            if len(image.shape) == 4:  # Shape: (C, D, H, W)
                central_slice_idx = image.shape[1] // 2  # Get depth slice index
                t1ce_slice = image[0, central_slice_idx, :, :]  # T1CE modality
                t2_slice = image[1, central_slice_idx, :, :]    # T2 modality
                flair_slice = image[2, central_slice_idx, :, :] # FLAIR modality
            else:
                raise ValueError("Unexpected tensor shape. Expected (C, D, H, W).")

            # Extract the corresponding segmentation mask slice
            central_slice_idx_mask = mask.shape[0] // 2  # For 3D mask (D, H, W)
            mask_slice = mask[central_slice_idx_mask, :, :]

            # Plot the T1CE, T2, FLAIR slices and the corresponding segmentation mask
            fig, axs = plt.subplots(1, 4, figsize=(20, 6))
            axs[0].imshow(t1ce_slice, cmap='gray')
            axs[0].set_title("T1CE Central Slice (Train)")
            axs[0].axis("off")

            axs[1].imshow(t2_slice, cmap='gray')
            axs[1].set_title("T2 Central Slice (Train)")
            axs[1].axis("off")

            axs[2].imshow(flair_slice, cmap='gray')
            axs[2].set_title("FLAIR Central Slice (Train)")
            axs[2].axis("off")

            axs[3].imshow(mask_slice, cmap=cmap, vmin=0, vmax=len(tumor_colors) - 1)
            axs[3].set_title("Segmentation Mask (Train)")
            axs[3].axis("off")

            plt.show()
            train_count += 1
            if train_count == 5:
                break
        if train_count == 5:
            break

    # Visualize 1 validation sample
    print("Visualizing 1 Validation Sample with Tumor Masks:")
    for images, masks in val_loader:
        image = images[0].cpu().numpy()
        mask = masks[0].cpu().numpy()

        # Check tensor shape and extract the central slice for each modality
        if len(image.shape) == 4:  # Shape: (C, D, H, W)
            central_slice_idx = image.shape[1] // 2  # Get depth slice index
            t1ce_slice = image[0, central_slice_idx, :, :]  # T1CE modality
            t2_slice = image[1, central_slice_idx, :, :]    # T2 modality
            flair_slice = image[2, central_slice_idx, :, :] # FLAIR modality
        else:
            raise ValueError("Unexpected tensor shape. Expected (C, D, H, W).")

        # Extract the corresponding segmentation mask slice
        central_slice_idx_mask = mask.shape[0] // 2  # For 3D mask (D, H, W)
        mask_slice = mask[central_slice_idx_mask, :, :]

        # Plot the T1CE, T2, FLAIR slices and the corresponding segmentation mask
        fig, axs = plt.subplots(1, 4, figsize=(20, 6))
        axs[0].imshow(t1ce_slice, cmap='gray')
        axs[0].set_title("T1CE Central Slice (Validation)")
        axs[0].axis("off")

        axs[1].imshow(t2_slice, cmap='gray')
        axs[1].set_title("T2 Central Slice (Validation)")
        axs[1].axis("off")

        axs[2].imshow(flair_slice, cmap='gray')
        axs[2].set_title("FLAIR Central Slice (Validation)")
        axs[2].axis("off")

        axs[3].imshow(mask_slice, cmap=cmap, vmin=0, vmax=len(tumor_colors) - 1)
        axs[3].set_title("Segmentation Mask (Validation)")
        axs[3].axis("off")

        plt.show()
        break

# %%

visualize_samples_with_colors(train_loader, val_loader)



# %%

class UNet3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(UNet3D, self).__init__()

        # Encoder path: Feature extraction
        self.encoder1 = self.conv_block(in_channels, 8)
        self.encoder2 = self.conv_block(8, 16)
        self.encoder3 = self.conv_block(16, 32)
        self.encoder4 = self.conv_block(32, 64)

        # Bottleneck layer
        self.bottleneck = self.conv_block(64, 128)

        # Decoder path: Feature reconstruction
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(128, 64)

        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(64, 32)

        self.upconv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(32, 16)

        self.upconv1 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(16, 8)

        # Final layer for output
        self.final_conv = nn.Conv3d(8, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """3D Convolution block with ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Define forward pass with skip connections."""
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2))

        # Decoder path
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))

        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        # Final output
        output = self.final_conv(dec1)
        return output

# Instantiate and test the model
model = UNet3D(in_channels=3, num_classes=4)
input_tensor = torch.randn(1, 3, 128, 128, 128)  # Batch size = 1, 3 channels, 128x128x128
output = model(input_tensor)
print(output.shape)  # Expected output: [1, 4, 128, 128, 128]


# %%

def compute_metrics(outputs, labels, num_classes):
    # Apply softmax to the model's raw output (logits)
    outputs = torch.nn.functional.softmax(outputs, dim=1)

    # Get the predicted class by taking the argmax over the softmax probabilities
    predictions = torch.argmax(outputs, dim=1)

    # Flatten the predictions and labels to 1D arrays
    true_labels = labels.cpu().numpy().flatten()
    predicted_labels = predictions.cpu().numpy().flatten()

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))

    # Initialize metrics dictionaries
    precision = {}
    recall = {}
    specificity = {}
    f1 = {}
    dice = {}
    accuracy = 0.0

    # Calculate per-class confusion matrix and metrics
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        precision[i] = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall[i] = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity[i] = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0.0
        dice[i] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0

    # Global accuracy
    accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'dice': dice
    }

# %%

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model setup
model = UNet3D(in_channels=3, num_classes=4)
model = nn.DataParallel(model)  # Multi-GPU training
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects integer labels
optimizer = optim.Adam(model.parameters(), lr= 0.001)
scaler = torch.amp.GradScaler(device.type)

num_epochs = 45
grad_accum_steps = 4  # Number of gradient accumulation steps
# early_stop_patience = 5  # Number of epochs to wait before stopping
# best_val_loss = float('inf')
# patience_counter = 0


# %%


# Training loop
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    running_specificity = 0.0
    running_dice = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Use autocast without the device_type argument
        with torch.autocast(device_type="cuda"):
            # with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):

            outputs = model(images)

            if outputs is None:
                continue

            # Calculate the loss and metrics
            loss = criterion(outputs, labels)
            num_classes = 4
            # You can include metrics here (accuracy, precision, recall, etc.)
            # Assuming compute_metrics() is defined elsewhere to calculate these metrics
            metrics = compute_metrics(outputs, labels, num_classes)

            # Scaler updates
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()

            # # Update running metrics
            running_loss += loss.item()

            # Average per-class metrics
            running_accuracy += metrics['accuracy']
            running_precision += np.mean(list(metrics['precision'].values()))  # Average precision for all classes
            running_recall += np.mean(list(metrics['recall'].values()))  # Average recall for all classes
            running_f1 += np.mean(list(metrics['f1'].values()))  # Average F1-score for all classes
            running_specificity += np.mean(list(metrics['specificity'].values()))  # Average specificity for all classes
            running_dice += np.mean(list(metrics['dice'].values()))  # Average Dice coefficient for all classes
            # Clear GPU memory periodically to avoid memory fragmentation
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.cuda.empty_cache()  # Clear cache to avoid fragmentation

    # Return average metrics for the epoch
    return (running_loss / len(train_loader),
            running_accuracy / len(train_loader),
            running_precision / len(train_loader),
            running_recall / len(train_loader),
            running_f1 / len(train_loader),
            running_specificity / len(train_loader),
            running_dice / len(train_loader))


# Update validate_one_epoch similarly, ensuring no device_type in autocast()
def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    running_specificity = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Use autocast without the device_type argument
            with torch.autocast(device_type="cuda"):
                outputs = model(images)

                if outputs is None:
                    continue

                # Calculate the loss and metrics
                loss = criterion(outputs, labels)
                num_classes = 4
                metrics = compute_metrics(outputs, labels, num_classes)

            # Update running metrics
            running_loss += loss.item()
            running_accuracy += metrics['accuracy']
            running_precision += np.mean(list(metrics['precision'].values()))  # Average precision for all classes
            running_recall += np.mean(list(metrics['recall'].values()))  # Average recall for all classes
            running_f1 += np.mean(list(metrics['f1'].values()))  # Average F1-score for all classes
            running_specificity += np.mean(list(metrics['specificity'].values()))  # Average specificity for all classes
            running_dice += np.mean(list(metrics['dice'].values()))  # Average Dice coefficient for all classes

    # Return average metrics for the epoch
    return (running_loss / len(val_loader),
            running_accuracy / len(val_loader),
            running_precision / len(val_loader),
            running_recall / len(val_loader),
            running_f1 / len(val_loader),
            running_specificity / len(val_loader),
            running_dice / len(val_loader))



# %%


# Initialize lists to track metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Directory to save outputs in Kaggle
output_dir = '/mnt/SSD2/archive'
os.makedirs(output_dir, exist_ok=True)

# Initialize an empty DataFrame to store metrics
metrics_df = pd.DataFrame(columns=[
    "Epoch",
    "Train Loss", "Validation Loss",
    "Train Accuracy", "Validation Accuracy",
    "Train Precision", "Validation Precision",
    "Train Recall", "Validation Recall",
    "Train F1", "Validation F1",
    "Train Specificity", "Validation Specificity",
    "Train Dice", "Validation Dice"
])

# Training loop with tqdm
print("Tracking epochs with tqdm...")
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    # Training phase
    train_loss, train_accuracy, train_precision, train_recall, train_f1, train_specificity, train_dice = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps
    )

    # Validation phase
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_specificity, val_dice = validate_one_epoch(
        model, val_loader, criterion, device
    )

    # Track metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Add metrics to the DataFrame (using pd.concat instead of append)
    new_row = pd.DataFrame({
        "Epoch": [epoch + 1],
        "Train Loss": [train_loss],
        "Validation Loss": [val_loss],
        "Train Accuracy": [train_accuracy],
        "Validation Accuracy": [val_accuracy],
        "Train Precision": [train_precision],
        "Validation Precision": [val_precision],
        "Train Recall": [train_recall],
        "Validation Recall": [val_recall],
        "Train F1": [train_f1],
        "Validation F1": [val_f1],
        "Train Specificity": [train_specificity],
        "Validation Specificity": [val_specificity],
        "Train Dice": [train_dice],
        "Validation Dice": [val_dice]
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # Print metrics for both training and validation
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, "
          f"Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Train Specificity: {train_specificity:.4f}, "
          f"Train Dice: {train_dice:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val Specificity: {val_specificity:.4f}, "
          f"Val Dice: {val_dice:.4f}")

    # Save checkpoint if validation loss improves
    if epoch+1 == num_epochs:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_3D_UNET.pth'))
        torch.save(model, os.path.join(output_dir, "entire_model_3D_UNET.pth"))
        print("Saved best model checkpoint.")

print("Training completed.")

# Save metrics DataFrame as a CSV file
csv_path = os.path.join(output_dir, 'training_metrics_3D_UNET.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"Training metrics saved to {csv_path}.")

# Plotting the loss and accuracy
print("Plotting and saving graphs...")

plt.figure(figsize=(14, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o', color="green")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o', color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", marker='o', color="green")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", marker='o', color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()

plt.tight_layout()

# Save the plot
graph_path = os.path.join(output_dir, 'training_graph_3D_UNET.png')
plt.savefig(graph_path)
plt.show()

print(f"Graph saved at {graph_path}.")