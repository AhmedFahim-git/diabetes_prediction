#!/bin/bash

docker compose up -d --build
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e predictions
sleep 100
pytest tests/
docker compose down
deactivate
