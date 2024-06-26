from pydantic import BaseSettings

ip = "145.97.18.149"
port = "7899"
db_file = "whisky"

class Settings(BaseSettings):
    DATABASE_IP: str = f"http://{ip}:{port}"
    DATABASE_FILE: str = f"{db_file}.db"
    


