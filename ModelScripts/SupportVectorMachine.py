import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

current_dir = os.path.dirname(__file__)  # Directory where the script is located
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Go one directory up
cleaned_emails_path = os.path.join(project_dir, "cleaned_emails_nn")
test_dir = os.path.join(cleaned_emails_path, "test_emails")  # Directory for test emails

# List to store email content for training
email_data = []

# Load all emails except "test-emails" for training
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

# List to store email content for testing
test_data = []

# Function to load emails from a folder (including subfolders)
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

# Load emails from the test directory
load_emails_from_folder(test_dir)

# Create DataFrames for training and testing
train_df = pd.DataFrame(email_data, columns=['email_text', 'label'])
test_df = pd.DataFrame(test_data, columns=['email_text', 'label'])

# Assign labels using spam keywords
spam_keywords = [
    'win', 'prize', 'free', 'cash', 'exclusive', 'lucky', 'lottery', 'selected', 'reward', 'won',
    'iwon', 'jackpot', 'buy', 'coupon', 'discount', 'romantic', 'www.match.com',
    'gifts', 'enter to win', 'complete a short survey', 'hotwebcash', 'sending spam', '100% approved',
    'no credit check', 'no late charges', 'apply here', 'eligible singles for you'
]
train_df['label'] = train_df['email_text'].apply(
    lambda x: 1 if any(word in x.lower() for word in spam_keywords) else 0
)
test_df['label'] = test_df['email_text'].apply(
    lambda x: 1 if any(word in x.lower() for word in spam_keywords) else 0
)

# Verify class distributions
print("\nTraining Class Distribution:")
print(train_df['label'].value_counts())

print("\nTesting Class Distribution:")
print(test_df['label'].value_counts())

# Vectorize the email text
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train = vectorizer.fit_transform(train_df['email_text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['email_text'])
y_test = test_df['label']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build the SVM model
svm_model = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced')
svm_model.fit(X_train_resampled, y_train_resampled)

# Predict the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: %.2f" % accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Raw):")
print(cm)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Normalize Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print("\nNormalized Confusion Matrix:")
print(cm_normalized)

# Plot Normalized Confusion Matrix
disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Not Spam", "Spam"])
disp_normalized.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Normalized)")
plt.show()

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nRecall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# K-Fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
print("\nK-Fold Results for SVM:")

# Perform K-Fold Cross-Validation
for train_index, validate_index in kf.split(X_train, y_train):
    # Split the data into training and validation sets
    X_train_fold = X_train[train_index]
    X_validate_fold = X_train[validate_index]
    y_train_fold = y_train.iloc[train_index]
    y_validate_fold = y_train.iloc[validate_index]

    # Train the SVM
    svm_model.fit(X_train_fold, y_train_fold)

    # Validate the model
    y_pred_fold = svm_model.predict(X_validate_fold)

    # Evaluate performance
    fold_accuracy = accuracy_score(y_validate_fold, y_pred_fold)
    print(f"Fold #{fold}: Accuracy: {fold_accuracy:.2f}")

    fold += 1
