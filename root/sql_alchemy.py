# database_manager.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sql_alchemy.models import Base

class SQLAlchemyManager:
    def __init__(self, database_url, echo=True):
        self.engine = create_engine(database_url, echo=echo)
        self.Session = sessionmaker(bind=self.engine)
        self.init_db()

    def init_db(self):
        """Initialize the database tables based on Base metadata."""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """Provide a session for use with a context manager or elsewhere."""
        return self.Session()

    def close_session(self, session):
        """Close the session if it's still open."""
        session.close()
