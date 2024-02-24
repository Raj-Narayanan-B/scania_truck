#FROM python:3.11.5
FROM apache/airflow:latest-python3.11
USER root
# COPY requirements.txt .
# COPY setup.py .
# RUN mkdir /app
# RUN mkdir -p /app/airflow
RUN apt-get update && apt-get install -y libgomp1
COPY . .
# WORKDIR /
# RUN pip cache purge
USER airflow
# chmod ug+w /opt/airflow/Secrets/Bundles
# RUN chmod -R 777 airflow-webserver
# RUN useradd -r airflow
# RUN chown -R airflow:0 /opt/airflow/Secrets
# RUN chmod -R 777 airflow-scheduler
# RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

# ENV AIRFLOW_HOME="/app/airflow"
# ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
# ENV AIRFLOW__CORE__DAGS_FOLDER="${AIRFLOW_HOME}/dags"
# ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='sqlite:////app/airflow/airflow.db'
# ENV AIRFLOW__DATABASE__SQL_ALCHEMY_SCHEMA="airflow_"
# RUN airflow init
# # RUN airflow db migrate
# RUN airflow users create -e brajnarayanan.b@gmail.com -f Raj -l Narayanan -p qwerty12345 -r Admin -u raj
# RUN chmod 777 start.sh
# RUN apt get-update -y
# ENTRYPOINT [ "/bin/sh" ]
# CMD ["start.sh"]

# sqlite:////app/airflow/airflow.db
# mysql+mysqldb://root:qwerty12345@localhost:3306/airflow_
# mysql+mysqldb://raj:admin@localhost:3306/airflow_db

#RUN apt update -y # && apt install awscli -y