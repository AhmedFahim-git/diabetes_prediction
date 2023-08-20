FROM python:3.9-slim
WORKDIR /home
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py script.sh diabetes-vid.csv aws_config aws_credentials ./
ENV MLFLOW_S3_ENDPOINT_URL=http://localstack:4566
ENV MLFLOW_TRACKING_URI=http://localhost:5000
RUN chmod +x ./script.sh
EXPOSE 8000
EXPOSE 5000
EXPOSE 4200
CMD ["./script.sh"]
