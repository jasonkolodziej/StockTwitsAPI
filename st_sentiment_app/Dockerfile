# Here we are creating an image for python alphine image.(https://hub.docker.com/r/library/python/)
FROM continuumio/miniconda3
# FROM python3

# Copying the requirements.txt first to leverage Docker cache
COPY ./st_sentiment_app/environment.yml /app/environment.yml
# COPY ./requirements.txt /app/requirements.txt


# WORKDIR is nothing but current directory (cd app)
WORKDIR /app

# Install the requirements in the current directory.
RUN conda env create -f environment.yml
# RUN pip install -r requirements.txt

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

# Copying the entire application to the docker container in the app directory.
COPY ./st_sentiment_app /app

# Setting environmental path to app directory. path environment variables tells shell,
# which directories to search for executable files.
ENV PATH /app:$PATH

# It executes the command python app.py in the app directory.
# start gunicorn
CMD ["gunicorn","--config","/app/gunicorn_config.py","app:app"]

EXPOSE 8005
