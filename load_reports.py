import requests
import os

# GitHub repository URL
repository_url = "https://api.github.com/repos/nandhishtr/DiseasePredictionLLM/contents/dataset_folder"

# Current working directory
current_directory = os.getcwd()

# Directory to store the downloaded files
download_directory = os.path.join(current_directory, "health_reports")

# Create the 'health_reports' directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)

# Make a GET request to the GitHub API to get the contents of the directory
response = requests.get(repository_url)
data = response.json()

# Iterate over each item (file or directory) in the directory
for item in data:
    # Check if the item is a file
    if item["type"] == "file":
        # Get the download URL for the file
        download_url = item["download_url"]

        # Get the filename
        filename = item["name"]

        # Check if the file is a text file (you can adjust this condition as needed)
        if filename.endswith(".txt"):
            # Make a GET request to download the file
            file_content = requests.get(download_url).content

            # Write the file content to a local file in the 'health_reports' directory
            with open(os.path.join(download_directory, filename), "wb") as f:
                f.write(file_content)

print("Text files downloaded successfully into the 'health_reports' folder!")
