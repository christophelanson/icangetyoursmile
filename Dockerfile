
# write some code to build your image
FROM python:3.8.6-buster

COPY 2Kimages_150epochs /2Kimages_150epochs
COPY icangetyoursmile /icangetyoursmile
COPY api /api
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
