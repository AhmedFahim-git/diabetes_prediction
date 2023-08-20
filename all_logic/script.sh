#!/bin/bash

mkdir -p ~/.aws
mv aws_config ~/.aws/config
mv aws_credentials ~/.aws/credentials
python make_bucket.py
mlflow server --backend-store-uri postgresql://user:example@db:5432/my_db --default-artifact-root s3://my-bucket --host 0.0.0.0 &
sleep 2
prefect server start --host 0.0.0.0 &
sleep 2
prefect block register --file block.py
sleep 2
prefect deployment build model.py:main_flow --name my_deployment --infra process --skip-upload  --apply -sb local-file-system/mylocal
prefect deployment run 'main-flow/my_deployment'
sleep 2
prefect agent start -q 'default' &
sleep 10
uvicorn deploy:app --host 0.0.0.0
