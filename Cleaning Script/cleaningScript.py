import os
import csv
import re

# Function to clean the email body by removing common footer patterns
def remove_footer(email_body):
    footer_patterns = [
        r"(?i)(best regards|sincerely|kind regards|thanks|regards|cheers|thank you)[\s\S]+",
        r"(?i)(unsubscribe|opt-out|email preferences|privacy policy)[\s\S]+",
        r"(?i)(contact details|company information)[\s\S]+",
        r"(?i)(confidentiality notice)[\s\S]+",
    ]
    for pattern in footer_patterns:
        email_body = re.sub(pattern, "", email_body)
    return email_body.strip()

# Function to read the raw email file and extract relevant fields
def extract_email_fields(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        email_content = file.read()

    date, sender, recipient, subject, body = None, None, None, None, None
    lines = email_content.split("\n")
    for line in lines:
        if line.lower().startswith("date:"):
            date = line[len("Date:"):].strip()
        elif line.lower().startswith("from:"):
            sender = line[len("From:"):].strip()
        elif line.lower().startswith("to:"):
            recipient = line[len("To:"):].strip()
        elif line.lower().startswith("subject:"):
            subject = line[len("Subject:"):].strip()

    try:
        body_start = email_content.index("\n\n")
        body = email_content[body_start:].strip()
    except ValueError:
        body = ""

    body_cleaned = remove_footer(body)

    return {
        "Date": date,
        "From": sender,
        "To": recipient,
        "Subject": subject,
        "Message": body_cleaned
    }

# Function to ensure that the directory for the CSV file exists
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Function to save data to a CSV file
def save_to_csv(data, csv_file_path):
    ensure_directory_exists(os.path.dirname(csv_file_path))
    if not data:
        print("No email data to save.")
        return
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Date", "From", "To", "Subject", "Message"])
        writer.writeheader()
        writer.writerows(data)

# Function to process multiple email files from a directory
def process_emails_from_directory(directory_path, output_directory, max_files=50):
    # Get a list of all files in the directory
    email_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    
    num_files = min(len(email_files), max_files)  # Process up to max_files or all files if fewer
    for idx, file_path in enumerate(email_files[:num_files]):
        try:
            email_data = extract_email_fields(file_path)
            csv_file_name = f"{idx + 1}.csv"
            csv_file_path = os.path.join(output_directory, csv_file_name)
            save_to_csv([email_data], csv_file_path)
            print(f"Cleaned email {idx + 1} saved to {csv_file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Directory containing email files (replace with the actual directory path)
email_directory_path = r"maildir/zufferli-j/inbox"

# Output directory for cleaned emails
output_directory = r"cleaned_emails/zufferli-j"

# Process emails from the directory
process_emails_from_directory(email_directory_path, output_directory)
