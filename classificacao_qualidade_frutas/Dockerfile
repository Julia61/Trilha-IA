FROM python:3.12

WORKDIR /code/app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
