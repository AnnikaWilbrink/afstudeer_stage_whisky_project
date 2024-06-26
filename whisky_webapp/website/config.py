from pydantic import BaseSettings

ip = "145.97.18.149"
port = "7899"

class Settings(BaseSettings):
    DATABASE_URL: str = f"http://{ip}:{port}"
    DATA_PATH: str = "/exports/nas/wilbrink.a/whisky_project/whisky_data/training_whisky_tics/"
    


