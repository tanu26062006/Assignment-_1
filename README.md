import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
import kagglehub
from sklearn.model_selection import train_test_split # Make sure this is imported at the top

# Download latest version
base_path = kagglehub.dataset_download("berlinsweird/devanagari")

# The `base_path` now points to the root directory where the dataset contents are.
# For example, it might look like '/root/.cache/kagglehub/datasets/berlinsweird/devanagari/versions/1'.
# Inside this, you'd typically find 'DevanagariHandwrittenCharacterDataset'.
# So, the actual path to your 'Train' and 'Test' folders will be nested.
# Let's verify the actual structure.
print(f"Path to downloaded dataset: {base_path}")

# Check what's inside base_path to confirm the structure
# print(os.listdir(base_path)) # This might show 'DevanagariHandwrittenCharacterDataset'
# Assuming the structure is base_path/DevanagariHandwrittenCharacterDataset/Train and Test
# You need to adjust base_data_dir to point to the directory containing 'Train' and 'Test'

# Let's find the correct root directory containing 'Train' and 'Test'
# The dataset on Kaggle has a folder 'DevanagariHandwrittenCharacterDataset' inside the archive.
# So, after unzipping/downloading with kagglehub, `base_path` will contain this folder.
# We need to construct the path to THIS folder.

# Use a more robust way to find the actual data root
actual_data_root = None
for root, dirs, files in os.walk(base_path):
    if 'Train' in dirs and 'Test' in dirs:
        actual_data_root = root
        break

if actual_data_root is None:
    print("Error: Could not find 'Train' and 'Test' directories within the downloaded dataset.")
    # You might need to manually inspect the structure if this error occurs
    # For common Kaggle dataset structures, it's often directly under base_path or one level deeper.
    # A common structure would be: base_path/DevanagariHandwrittenCharacterDataset/Train and Test
    # So, we can try to assume this:
    if os.path.exists(os.path.join(base_path, 'DevanagariHandwrittenCharacterDataset')):
        actual_data_root = os.path.join(base_path, 'DevanagariHandwrittenCharacterDataset')
    else:
        # Fallback to base_path if no specific nested folder found
        actual_data_root = base_path


print(f"Actual data root (containing Train/Test): {actual_data_root}")


def load_devanagari_data(data_root_path): # Renamed base_path to data_root_path for clarity
    images = []
    labels = []

    train_path = os.path.join(data_root_path, 'Train')
    test_path = os.path.join(data_root_path, 'Test')

    # Verify that Train and Test directories exist
    if not os.path.isdir(train_path):
        print(f"Error: Train directory not found at {train_path}")
        return None, None, None
    if not os.path.isdir(test_path):
        print(f"Error: Test directory not found at {test_path}")
        return None, None, None

    # Get class names from the 'Train' directory
    class_names = sorted(os.listdir(train_path))
    class_names = [d for d in class_names if os.path.isdir(os.path.join(train_path, d))] # Ensure it's a directory
    label_map = {name: i for i, name in enumerate(class_names)}

    print(f"Detected {len(class_names)} classes: {class_names}")

    for phase in ['Train', 'Test']:
        phase_path = os.path.join(data_root_path, phase)
        print(f"Loading images from: {phase_path}")
        for char_folder in os.listdir(phase_path):
            char_path = os.path.join(phase_path, char_folder)
            if os.path.isdir(char_path) and char_folder in label_map: # Ensure it's a directory and a known class
                label = label_map[char_folder]
                for img_name in os.listdir(char_path):
                    img_path = os.path.join(char_path, img_name)
                    # Use os.path.join to construct full path reliably
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (32, 32)) # Resize to a common size
                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"Warning: Could not read image {img_path}")
    return np.array(images), np.array(labels), class_names

# Pass the correctly identified data root to your loading function
X, y, class_names = load_devanagari_data(actual_data_root)

if X is None:
    print("Data loading failed. Exiting.")
    exit() # Exit if data loading failed

print(f"Total images loaded: {len(X)}")
print(f"Total labels loaded: {len(y)}")

# Normalize images to 0-1 range
X = X / 255.0

# Reshape images for CNN input (batch, height, width, channels)
X = X.reshape(-1, 32, 32, 1) # Assuming grayscale, so 1 channel

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(class_names))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Step 2: Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Print model summary to verify layers and parameters

# Step 4: Train the model
epochs = 20 # Increased epochs for potentially better training
history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test))

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Optional: Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
