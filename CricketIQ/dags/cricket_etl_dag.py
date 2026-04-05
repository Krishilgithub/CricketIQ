from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'cricketiq',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'cricketiq_etl_pipeline',
    default_args=default_args,
    description='A simple ETL DAG for CricketIQ',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['cricketiq'],
) as dag:

    # 1. Simulate Live Data Drop
    # Generates a JSON file in data/raw/new_json_drops
    simulate_live_data = BashOperator(
        task_id='simulate_live_data',
        bash_command='python /opt/airflow/scripts/simulate_realtime_data.py'
    )

    # 2. Ingest Live Data (Incremental)
    # Reads the drop folder and appends the new match to DuckDB Bronze 
    ingest_live_data = BashOperator(
        task_id='ingest_live_data',
        bash_command='python -m src.ingestion.ingest_live_json'
    )

    # 3. Run dbt models (Silver layer)
    run_dbt_silver = BashOperator(
        task_id='run_dbt_silver',
        bash_command='cd /opt/airflow/dbt && dbt build --select silver'
    )

    # 4. Run dbt models (Gold layer)
    run_dbt_gold = BashOperator(
        task_id='run_dbt_gold',
        bash_command='cd /opt/airflow/dbt && dbt build --select gold'
    )

    # 5. Feature Engineering
    run_feature_engineering = BashOperator(
        task_id='run_feature_engineering',
        bash_command='python /opt/airflow/src/features/feature_engineering.py'
    )

    # 6. Model Retraining (Optional/Monitoring)
    run_model_retraining = BashOperator(
        task_id='run_model_retraining',
        bash_command='python /opt/airflow/src/monitoring/retrain_trigger.py'
    )

    # Define task dependencies
    simulate_live_data >> ingest_live_data >> run_dbt_silver >> run_dbt_gold >> run_feature_engineering >> run_model_retraining
