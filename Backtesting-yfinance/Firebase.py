import firebase_admin
from firebase_admin import credentials, db

class InitializeFirebase:
    def __init__(self):
        # Path to your Firebase Admin SDK JSON
        cred_path = 'trading-8a164-firebase-adminsdk-jq4mh-55b2278f8c.json'
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://trading-8a164-default-rtdb.firebaseio.com/'
        })
        self.db = db.reference()
