

## Repo structure

`preprocessing/` - Raw data clean and panel construction (on server Spark)

`modelling/` - Final data preprocessing, feature engineering and model fit (on server Spark + sklearn)

`models/` - Stored fitted models + features + inference of models from raw input from frontend

`backend/` - FAST API backend for OAS service (Docker)

`frontend/` - Streamlit frontend for OAS service (Docker)

`eda/` - EDA for the project

## How to run service

If you have PyCharm just play in `docker-compose.yml`

- Run following commands:
`docker compose build`
`docker compose up`

- Open: `http://localhost:8501`
- At the end run: `docker compose down`
