FROM bentoml/model-server:latest
COPY . /app
WORKDIR /app
RUN bentoml build
CMD ["bentoml", "serve", "bento_name:latest"]
