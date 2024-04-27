import os

import wget
from fastapi import requests

"""
def download_files(pdf_url, base_url, num_files):
    # Path where the PDF file will be saved
    pdf_file_path = "bcg-2022-annual-sustainability-report-apr-2023.pdf"

    # Download the PDF file
    response = requests.get(pdf_url)
    with open(pdf_file_path, 'wb') as f:
        f.write(response.content)

    # Loop through each file number and download it
    for i in range(num_files):
        # Construct the URL for each file
        file_url = base_url + f"health_report_{{{i}}}/health_report_{{{i}}}.txt"

        # Download the file
        response = requests.get(file_url)

        # Save the file
        with open(f"health_report_{i}.txt", 'wb') as f:
            f.write(response.content)


def clone_repository(repo_url, destination_path):
    # Clone the repository using Git
    os.system(f"git clone {repo_url} {destination_path}")


# Define the base URL where the health report files are located
base_url = "https://github.com/nandhishtr/DiseasePredictionLLM/raw/main/dataset_folder/"

# Define the URL of the PDF file to download
pdf_url = "https://github.com/karan-nanonets/llamaindex-guide/raw/main/bcg-2022-annual-sustainability-report-apr-2023.pdf"

# Define the number of health report files you want to download
num_files = 10  # For example, if you want to download 10 files

# Define the URL of the repository to clone
repo_url = "https://github.com/nandhishtr/DiseasePredictionLLM.git"

# Define the destination path where the repository will be cloned
destination_path = "DiseasePredictionLLM"

# Call the method to clone the repository
clone_repository(repo_url, destination_path)
# Call the method to download the files
download_files(pdf_url, base_url, num_files)

"""
import requests

# Define the base URL where the files are located
base_url = "https://github.com/nandhishtr/DiseasePredictionLLM/raw/main/dataset_folder/"

# Define the number of files you want to download
num_files = 75  # For example, if you want to download 10 files

# Loop through each file number and download it
for i in range(num_files):
    # Construct the URL for each file
    file_url = base_url + "health_report_%7B{}%7D/health_report_%7B{}%7D.txt".format(i, i)

    # Download the file
    response = requests.get(file_url)

    # Save the file
    with open(f"health_report_{i}.txt", 'wb') as f:
        f.write(response.content)
