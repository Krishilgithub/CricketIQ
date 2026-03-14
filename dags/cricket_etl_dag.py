from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Add project root to path
PROJECT_ROOT = Path("/opt/airflow/project") # Assuming dockerized airflow or mapped path
sys.path.insert(0, str(PROJECT_ROOT))

default_args = {
    'owner': 'cricket_iq',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 14),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'cricket_medallion_etl',
    default_args=default_args,
    description='ETL pipeline for CricketIQ transforming data from Bronze to Gold',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# We can use BashOperator to run the existing python scripts
# Make sure to set the environment paths correctly
project_dir = "/opt/airflow/project" if os.getenv("AIRFLOW_HOME") else str(Path(__file__).parent.parent)

t1_run_bronze = BashOperator(
    task_id='load_bronze_layer',
    bash_command=f"cd {project_dir} && python src/etl/run_pipeline.py --stage bronze",
    dag=dag,
)

t2_run_silver = BashOperator(
    task_id='load_silver_layer',
    bash_command=f"cd {project_dir} && python src/etl/run_pipeline.py --stage silver",
    dag=dag,
)

t3_run_gold = BashOperator(
    task_id='load_gold_layer',
    bash_command=f"cd {project_dir} && python src/etl/run_pipeline.py --stage gold",
    dag=dag,
)

t1_run_bronze >> t2_run_silver >> t3_run_gold
