# start_project.ps1
Write-Host "Starting CricketIQ Project Components..." -ForegroundColor Green

$project_dir = "D:\Projects\CricketIQ\CricketIQ"

# 1. Start MLflow Tracking Server
Write-Host "Starting MLflow Server on port 5050..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd $project_dir; .\.venv\Scripts\activate; mlflow ui --port 5050`""

# 2. Skip FastAPI Backend (Since this is a Streamlit App only)
Write-Host "Skipping FastAPI Backend (Not found)..." -ForegroundColor Cyan
# Start-Process powershell -ArgumentList "-NoExit -Command `"cd $project_dir; .\.venv\Scripts\activate; uvicorn src.app:app --reload`""

# 3. Start Streamlit Frontend
Write-Host "Starting Streamlit App on port 8501..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd $project_dir; .\.venv\Scripts\activate; `$env:PYTHONPATH='.'; streamlit run src/app.py --server.port 8501`""

# 4. Attempt to Start Airflow via Docker
Write-Host "Starting Apache Airflow Dashboard via Docker..." -ForegroundColor Cyan
# Re-running the init just in case it failed previously, then launching it
Start-Process powershell -ArgumentList "-NoExit -Command `"cd $project_dir; docker compose -f docker-compose.airflow.yml up airflow-init -d; docker compose -f docker-compose.airflow.yml up -d`""

Write-Host "All components have been launched in separate terminal windows." -ForegroundColor Green
Write-Host "Streamlit: http://localhost:8501"
Write-Host "FastAPI: http://localhost:8000"
Write-Host "MLflow: http://localhost:5050"
Write-Host "Airflow: http://localhost:8080 (Once Docker finishes pulling)"
