FROM python:3.10-slim

WORKDIR /app

COPY ./fastapi /app

RUN pip install --no-cache-dir fastapi uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
