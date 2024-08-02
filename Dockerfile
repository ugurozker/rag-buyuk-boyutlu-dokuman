FROM docker.io/python:3.12-slim as app
# RUN apt-get update --allow-unauthenticated -y
WORKDIR /usr/src/app
COPY requirements.txt ./
# use no-cache-dir to limit disk usage https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
RUN pip install -r requirements.txt --no-cache-dir

COPY . ./
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "main:app"]

EXPOSE 8000
