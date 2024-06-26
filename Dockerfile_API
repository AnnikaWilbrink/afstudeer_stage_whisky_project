FROM mambaorg/micromamba:0.27.0

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml

RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

WORKDIR /code

COPY --chown=$MAMBA_USER:$MAMBA_USER ./app /code/app

COPY --chown=$MAMBA_USER:$MAMBA_USER config.py /code/config.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7899", "--reload"]

