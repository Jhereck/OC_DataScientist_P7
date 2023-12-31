# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-bullseye

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update

# Install & use pipenv
COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --ignore-pipfile --dev --system --deploy

WORKDIR /app
COPY . /app

RUN pytest

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
EXPOSE 9000
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "9000", "app.api:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--conf", "app/gunicorn_conf.py", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]