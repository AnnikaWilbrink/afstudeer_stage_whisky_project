from sqlmodel import SQLModel, create_engine
import os
from config import Settings

app_settings = Settings()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
conn_str = 'sqlite:///' + os.path.join(BASE_DIR, f'{app_settings.DATABASE_FILE}')

DATABASE_URL=os.getenv("DATABASE_URL", conn_str)

engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

