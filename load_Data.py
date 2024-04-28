import requests

url = "https://raw.githubusercontent.com/nandhishtr/DiseasePredictionLLM/main/dataset_folder/health_report_{0}/health_report_{0}.txt"
response = requests.get(url)

if response.status_code == 200:
    with open("health_report.txt", "wb") as f:
        f.write(response.content)
        print("File downloaded successfully.")
else:
    print("Failed to download file.")
