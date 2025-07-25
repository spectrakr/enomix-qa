import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory settings
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"
STATIC_DIR = "./static"
LOG_DIR = "./logs"

# Create directories if they don't exist
for directory in [CHROMA_DIR, DOCS_DIR, STATIC_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Logging configuration
LOG_FILE = os.path.join(LOG_DIR, "app.log") 