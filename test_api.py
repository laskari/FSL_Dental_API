import requests
import time
url = "http://localhost:8002/ada_extraction"

# Define the payload data as a dictionary
data = {
   # "FilePath": r"/Data/FSL_prod_codebase/FSL_codebase/api/ADA/images/2025I4A9D015_001.tiff"
    "FilePath": r"D:\project\FSL\FSL_codebase\api\ADA\images\2027C43AD003_001.jpg"
}

try:
    # Measure the start time
    start_time = time.time()

    # Make the POST request with the JSON payload
    response = requests.post(url, json=data)

    # Measure the total time taken for execution
    total_time_taken = time.time() - start_time
    print(f"Total time taken for execution: {total_time_taken}")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        print("Response:", response.json())
        print("Request was successful.")
    else:
        # If request was not successful, print error message
        print("Error:", response.content)
        print(f"Request failed with status code {response.status_code}")
except Exception as e:
    print("Error occurred during API call:", e)

# import requests

# # Define the URL where the endpoint is hosted
# url = "http://localhost:8080/ada_extraction"

# # Path to the file you want to upload
# file_path = r"D:\project\FSL\FSL_codebase\api\ADA\images\2027C43AD001_001.jpg"

# # Open the file in binary mode
# with open(file_path, "rb") as file:
#     # Prepare the file to be uploaded
#     files = {"file": file}

#     # Send the POST request with the file
#     response = requests.post(url, files=files)

# # Print the response
# print(response.json())
