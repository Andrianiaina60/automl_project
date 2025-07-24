
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
 && pip install mlflow scikit-learn pandas

EXPOSE 5000
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
