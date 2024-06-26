# Whisky Web Application
The web application analyzes whiskies with data from the database.

## Run Web Application Manually
1. To run the API, clone the whisky_api repository and use the following commands:
```
conda env create -f env.yml -n api
conda activate api
cd whisky_api
uvicorn app.main:app --host 0.0.0.0 --port 7899 --reload
```

2. Clone the whisky_webapp repository.

3. Create a conda environment from the env.yml and activate the new environment with the following commands:
```
conda env create -f env.yml -n whiskyapp
conda activate whiskyapp
```

4. Run the server with the commands:
```
cd whisky_webapp/website
python manage.py runserver 0:8000
```

## Run Web Application via Docker
1. Clone both the whisky_api and the whisky_webapp repository.

2. Move the docker-compose.yml from whisky_webapp to the parent directory:
```
mv whisky_webapp/docker-compose.yml docker-compose.yml
```
3. Add a database directory to the parent directory:
```
mkdir database
chmod -R 777 database
```
4. Add the sqlite database file (from whisky_api/app/) to the new database directory.

5. Run the docker-compose.yml with the command:
```
docker-compose up
```

#### NOTES
The config.py contains the uvicorn host and port that are used in the other scripts and can be changed in that file.
