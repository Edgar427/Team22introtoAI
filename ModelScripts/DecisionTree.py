import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

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
X_train = vectorizer.fit_transform(train_df['email_text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['email_text'])
y_test = test_df['label']

# Step 5: Handle Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 6: Decision Tree with Fixed Hyperparameters
decision_tree = DecisionTreeClassifier(
    criterion='gini', max_depth=20, min_samples_split=5,
    min_samples_leaf=5, class_weight='balanced')

# Train the Decision Tree
decision_tree.fit(X_train_resampled, y_train_resampled)

# Evaluate on Test Data
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nDecision Tree Accuracy on Test Data: %.2f" % accuracy)


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

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nRecall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Step 8: K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
print("\nK-Fold Results:")

# Perform K-Fold Cross-Validation
for train_index, validate_index in kf.split(X_train_resampled, y_train_resampled):
    # Split the data into training and validation sets
    X_train_fold = X_train_resampled[train_index]
    X_validate_fold = X_train_resampled[validate_index]
    y_train_fold = y_train_resampled[train_index]
    y_validate_fold = y_train_resampled[validate_index]

    # Train the Decision Tree
    decision_tree.fit(X_train_fold, y_train_fold)

    # Validate the model
    y_pred_fold = decision_tree.predict(X_validate_fold)

    # Evaluate performance
    fold_accuracy = accuracy_score(y_validate_fold, y_pred_fold)
    print(f"Fold #{fold}, Accuracy: {fold_accuracy:.2f}")
    fold += 1

