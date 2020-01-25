FROM python:3.6-slim
COPY requirements.pip requirements.pip
RUN pip install -r requirements.pip
RUN pip install imageai --upgrade
RUN apt-get update && apt-get install -y libglib2.0-0
EXPOSE 5000
WORKDIR /opt/app
ENTRYPOINT [ "python", "./app.py" ]