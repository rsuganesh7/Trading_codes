"""_summary_
login_manager.py
"""


from pyotp import TOTP
from SmartApi import SmartConnect
from Trading_codes.Live import config
class LoginManager:
    def __init__(self):
        print("THe login manager is created")
        self.api_key = config.API_KEY
        self.secret_code = config.SECRET_CODE
        self.client_code = config.CLIENT_CODE
        self.password = config.PASSWORD
        self.totp_key = config.TOTP_KEY
        self.obj = None
        self.feed_token = None
        self.user_profile = None
        self.data = None
        print(f'The config is {config.API_KEY}')

    def generate_session(self):
        """Generate a session using the loaded credentials."""
        try:
            self.obj = SmartConnect(api_key=self.api_key)
            totp = TOTP(self.totp_key).now()
            data = self.obj.generateSession(clientCode=self.client_code, password=self.password, totp=totp)
            self.data = data
        except Exception as e:
            raise ValueError(f"Failed to generate session: {e}")

    def get_feed_token(self):
        """Retrieve the feed token."""
        try:
            if self.obj:
                self.data['refresh_token'] = self.obj.getfeedToken()
            else:
                raise ValueError("Session not generated. Call generate_session() first.")
        except Exception as e:
            raise ValueError(f"Failed to get feed token: {e}")

    def get_user_profile(self):
        """Retrieve the user profile using the refresh token."""
        try:
            if self.obj and self.data['refresh_token']:
                self.user_profile = self.obj.getProfile(self.data['refresh_token'])
            else:
                raise ValueError(
                    "Session not generated or refresh token not available. Call generate_session() first."
                )
        except Exception as e:
            raise ValueError(f"Failed to get user profile: {e}")

    def login_and_retrieve_info(self) :
        """Convenience method to perform all steps: Generate session, get feed token, and user profile."""
        self.generate_session()
        self.get_feed_token()
        self.get_user_profile()
        return self.obj, self.data, self.user_profile

