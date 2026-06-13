import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay , precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure Deterministic Operations in TensorFlow
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Step 1: Load Data
current_dir = os.path.dirname(__file__)  # Directory where the script is located
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Go one directory up
cleaned_emails_path = os.path.join(project_dir, "cleaned_emails_nn")
test_dir = os.path.join(cleaned_emails_path, "test_emails")  # Directory for test emails

# List to store email content
email_data = []

# Load all emails except from "allen-p" for training
all_folders = [
    folder for folder in os.listdir(cleaned_emails_path)
    if folder != "test-emails" and not folder.startswith(".")  # Exclude hidden files like `.DS_Store`
]
for folder in all_folders:
    folder_path = os.path.join(cleaned_emails_path, folder)
    if os.path.isdir(folder_path):  # Ensure it's a directory
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        email_data.append([content, 0])  # Default label
                except Exception as e:
                    print(f"Could not read {file_name}: {e}")

# Load "allen-p" emails for testing
#allen_path = os.path.join(cleaned_emails_path, "allen-p")
test_data = []
test_data = []

def load_emails_from_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):  # Recursively walk through subfolders
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        test_data.append([content, 0])  # Default label for testing
                except Exception as e:
                    print(f"Could not read {file_name}: {e}")

# Load emails from the test directory (including subfolders)
load_emails_from_folder(test_dir)

# Step 2: Create DataFrames
train_df = pd.DataFrame(email_data, columns=['email_text', 'label'])
test_df = pd.DataFrame(test_data, columns=['email_text', 'label'])

# Step 3: Assign Labels Using Keywords
spam_keywords = [
    'win', 'prize', 'free', 'cash', 'exclusive', 'lucky', 'lottery', 'selected', 'reward', 'won',
    'iwon', 'jackpot', 'buy', 'coupon', 'discount', 'romantic', 'www.match.com',
    'gifts', 'enter to win', 'complete a short survey', 'hotwebcash', 'sending spam', '100% approved',
    'no credit check', 'no late charges', 'apply here', 'eligible singles for you'
]
train_df['label'] = train_df['email_text'].apply(lambda x: 1 if any(word in x.lower() for word in spam_keywords) else 0)
test_df['label'] = test_df['email_text'].apply(lambda x: 1 if any(word in x.lower() for word in spam_keywords) else 0)

print("\nTraining Class Distribution:")
print(train_df['label'].value_counts())

print("\nTesting Class Distribution:")
print(test_df['label'].value_counts())

# Step 4: Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train = vectorizer.fit_transform(train_df['email_text']).toarray()
y_train = train_df['label']
X_test = vectorizer.transform(test_df['email_text']).toarray()
y_test = test_df['label']

# Step 5: Handle Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 6: Neural Network Model with TensorFlow
model = Sequential([
    Input(shape=(X_train_resampled.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.02)),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.02)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Neural Network
#early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train_resampled, y_train_resampled, 
    epochs=300, batch_size=32, #11 is optimal
    validation_split=0.2, verbose=2,
    #callbacks=[early_stopping]
)

# Evaluate on Test Data
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_test, y_pred)
print("\nNeural Network Accuracy on Test Data: %.2f" % accuracy)

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Raw Numbers):")
print(cm)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam", "Spam"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print("\nNormalized Confusion Matrix (Proportions):")
print(cm_normalized)

# Plot Normalized Confusion Matrix
disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Not Spam", "Spam"])
disp_normalized.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Normalized)")
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
print("\nK-Fold Results:")

# Initialize lists to store metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Perform K-Fold Cross-Validation
for train_index, validate_index in kf.split(X_train_resampled, y_train_resampled):
    # Split the data into training and validation sets
    X_train_fold = X_train_resampled[train_index]
    X_validate_fold = X_train_resampled[validate_index]
    y_train_fold = y_train_resampled[train_index]
    y_validate_fold = y_train_resampled[validate_index]

    # Reinitialize the model for each fold
    model = Sequential([
        Input(shape=(X_train_resampled.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train_fold, y_train_fold,
        epochs=11, batch_size=16, verbose=0
    )

    # Validate the model
    y_pred_fold = (model.predict(X_validate_fold) > 0.5).astype(int).flatten()

    # Calculate Precision, Recall, and F1-Score for the current fold
    precision = precision_score(y_validate_fold, y_pred_fold)
    recall = recall_score(y_validate_fold, y_pred_fold)
    f1 = f1_score(y_validate_fold, y_pred_fold)
    fold_accuracy = accuracy_score(y_validate_fold, y_pred_fold)

    # Append results to lists for calculating mean and std
    accuracies.append(fold_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    # Print the results for the current fold
    print(f"Fold #{fold}, Accuracy: {fold_accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    fold += 1

# Calculate and print the mean and standard deviation across all folds
print("\nK-Fold Cross-Validation Average Metrics:")
print(f"Average Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Average Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Average Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"Average F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")