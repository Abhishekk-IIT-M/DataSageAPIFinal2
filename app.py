import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import traceback

# Assume this class will be our main agent logic file later
from data_analyst_agent import DataAnalystAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)
# A secret key is needed for Flask to handle some internal things
app.secret_key = os.environ.get("SESSION_SECRET",
                                "dev-secret-key-for-tds-project")

# Create a temporary directory for file uploads
UPLOAD_FOLDER = tempfile.mkdtemp()

# Initialize our Data Analyst Agent
# This object will contain all the core logic
data_agent = DataAnalystAgent()


# The project requires the endpoint to be at /api/
@app.route('/', methods=['POST'])
def analyze_data_endpoint():
    """
    Main endpoint for data analysis requests as per the project spec.
    It accepts a 'questions.txt' file and optional data files.
    """
    logger.info("Received request at /api/")

    # --- Simplified Input Handling ---
    if 'questions.txt' not in request.files:
        logger.error("Request missing 'questions.txt' file.")
        return jsonify(
            {"error": "Request must include a 'questions.txt' file."}), 400

    temp_files = {}  # To store paths of temporarily saved files
    try:
        # Save questions.txt and read its content
        questions_file = request.files['questions.txt']
        question_content = questions_file.read().decode('utf-8')
        logger.info(f"Question received: {question_content[:100]}...")

        # Save all other attached files temporarily
        for key, file in request.files.items():
            if key != 'questions.txt':
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                temp_files[key] = filepath
                logger.info(
                    f"Saved temporary data file: {filename} at {filepath}")

        # --- Simplified Agent Invocation ---
        # We will build the agent logic to handle this simple input
        result = data_agent.run(question=question_content, files=temp_files)

        logger.info("Analysis completed successfully.")
        # The agent is expected to return a JSON-serializable object (dict or list)
        return jsonify(result)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error":
                        f"An unexpected error occurred: {str(e)}"}), 500

    finally:
        # --- Cleanup ---
        # Ensure all temporary files are deleted after the request is handled
        for file_path in temp_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")


# Optional: Add a simple root route to confirm the server is running
@app.route('/')
def index():
    return "Data Analyst Agent API is running."
