# Whisky API and Pipeline
The whisky web application uses a database. This is created with an API. The pipeline then collects information from GC-MS data and populates the database via the API.

## Creation of sqlite database
1. Clone the whisky_api repository

### API (Manually)
2. Create a conda environment from the env.yml and activate the new environment with the following commands:
```
conda env create -f env.yml -n api
conda activate api
```

3. The database is created on startup of the API. While the API runs, other scripts can communicate with the API to get and post the contents of the database. The API starts by use of the commands:
```
cd whisky_api
uvicorn app.main:app --host 0.0.0.0 --port 7899 --reload
```

### API (via Docker)
2. Build an image from the dockerfile with the commands:
```
cd whisky_api
docker build --no-cache -t whisky_api .
```

3. Run the image with the command:
```
docker run -p 7899:7899 whisky_api
```

#### NOTES
The config.py contains the uvicorn host and port that are used in the other scripts and can be changed in that file.

## Populate database

### Snakemake
1. Clone the whisky_api repository if not already done

2. Create a conda environment, activate the environment and install snakemake:
```
conda create -n snakemake
conda activate snakemake
conda install -c bioconda snakemake
```

3. Run the following commands to start the pipeline:
```
cd whisky_api
snakemake --use-conda -c1 --snakefile Snakefile_whisky
```
#### NOTES
The config.yml contains all the paths that are needed and can be changed in that file.
Run the pipeline while the API runs.
