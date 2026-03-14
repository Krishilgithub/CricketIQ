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

    # 1. Ingest Data (Simulating the pull or parsing)
    # This could run the existing ingest_historical or a script
    ingest_live_data = BashOperator(
        task_id='ingest_live_data',
        bash_command='python /opt/airflow/src/ingestion/ingest_historical.py'
    )

    # 2. Run dbt models (Silver layer)
    run_dbt_silver = BashOperator(
        task_id='run_dbt_silver',
        bash_command='cd /opt/airflow/dbt && dbt build --select silver'
    )

    # 3. Run dbt models (Gold layer)
    run_dbt_gold = BashOperator(
        task_id='run_dbt_gold',
        bash_command='cd /opt/airflow/dbt && dbt build --select gold'
    )

    # 4. Feature Engineering
    run_feature_engineering = BashOperator(
        task_id='run_feature_engineering',
        bash_command='python /opt/airflow/src/features/feature_engineering.py'
    )

    # 5. Model Retraining (Optional/Monitoring)
    run_model_retraining = BashOperator(
        task_id='run_model_retraining',
        bash_command='python /opt/airflow/src/monitoring/retrain_trigger.py'
    )

    # Define task dependencies
    ingest_live_data >> run_dbt_silver >> run_dbt_gold >> run_feature_engineering >> run_model_retraining
