#config.py
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Load environment variables from your custom-named .env file
env = load_dotenv(dotenv_path)


# Load credentials from environment variables
API_KEY = os.environ.get('API_KEY')
SECRET_CODE = os.environ.get('SECRET_CODE')
CLIENT_CODE = os.environ.get('CLIENT_CODE')
PASSWORD = os.environ.get('PASSWORD')
TOTP_KEY = os.environ.get('TOTP_KEY')
DB_NAME=os.environ.get('DB_NAME')
DB_PASSWORD= os.environ.get('DB_PASSWORD')
DB_HOST= os.environ.get('DB_HOST')
DB_USER= os.environ.get('DB_USER')

print(f'TOTP key is {TOTP_KEY}')
# API Endpoints and Paths
INSTRUMENT_LIST_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
