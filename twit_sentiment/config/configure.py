# configure.py
## to try and automatically connect to user
from dotenv import load_dotenv
# explicitly providing path to '.env'
from pathlib import Path  # python3 only
import os
import requests
# load the environment 
try:
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path, verbose=True)
except:
    exit
connection = os.environ['GET_ACCESS_TOKEN']
print('connection was...')
print(connection)
print()
print('using default access token...')
# resp = requests.get(str(connection), timeout=5)
os.environ['ST_ACCESS_TOKEN'] = 'd13ac25e7597bf5c670042eb76739cee3e2cf78d'
# print(resp.headers)
