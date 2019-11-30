# Here we are creating an image for python alphine image.(https://hub.docker.com/r/library/python/)
FROM continuumio/miniconda3
# FROM python3

# Copying the requirements.txt first to leverage Docker cache
COPY ./st_ml/environment.yml /mlapi/environment.yml
# COPY ./requirements.txt /app/requirements.txt


# WORKDIR is nothing but current directory (cd app)
WORKDIR /mlapi

# Install the requirements in the current directory.
RUN conda env create -f environment.yml
# RUN pip install -r requirements.txt

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 ./environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 ./environment.yml | cut -d' ' -f2)/bin:$PATH

# Copying the entire application to the docker container in the app directory.
COPY ./st_ml /mlapi

# Setting environmental path to app directory. path environment variables tells shell,
# which directories to search for executable files.
ENV PATH /mlapi:$PATH

# It executes the command python app.py in the app directory.
# start gunicorn
CMD ["uvicorn","ml_api:app","--host","0.0.0.0","--port", "8675"]

EXPOSE 8675