import requests
import os

# The URL for the server running inside the same container
# Flask runs on port 5000 by default in your main.py
API_URL = "http://localhost:5000/api/"

# The path to the questions file in this project
QUESTIONS_FILE_PATH = 'questions.txt'


def run_test():
    """
    Sends a test request to the running Flask API.
    """
    print(f"--- Preparing to send request to {API_URL} ---")

    if not os.path.exists(QUESTIONS_FILE_PATH):
        print(
            f"ERROR: '{QUESTIONS_FILE_PATH}' not found. Please make sure the file exists."
        )
        return

    # Prepare the file for uploading
    with open(QUESTIONS_FILE_PATH, 'rb') as f:
        files = {'questions.txt': f}

        print("--- Sending POST request with questions.txt... ---")
        try:
            # Send the request
            response = requests.post(API_URL, files=files,
                                     timeout=180)  # 3 minute timeout

            # Print the results
            print(
                f"\n--- Response Received (Status Code: {response.status_code}) ---"
            )

            # Try to print the JSON response, or print the raw text if it's not JSON
            try:
                print("Response JSON:")
                print(response.json())
            except requests.exceptions.JSONDecodeError:
                print("Response Text (not valid JSON):")
                print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"\n--- An error occurred during the request ---")
            print(e)


if __name__ == "__main__":
    run_test()
