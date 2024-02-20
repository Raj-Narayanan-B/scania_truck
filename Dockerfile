FROM python:3.11.5
USER root
RUN mkdir /app
RUN mkdir -p /app/airflow
COPY . /app/
WORKDIR /app/
RUN pip cache purge
RUN pip install -r requirements.txt
ENV AIRFLOW__HOME = /app/airflow
ENV AIRFLOW__CORE__DAGBAG__IMPORT__TIMEOUT = 1000
ENV AIRFLOW__CORE__ENABLE__XCOM__PICKLING = True
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////app/airflow/airflow.db
RUN airflow db migrate
RUN airflow users create -e brajnarayanan.b@gmail.com -f Raj -l Narayanan -p qwerty12345 -r Admin -u raj
RUN chmod 777 start.sh
RUN apt update -y && apt install awscli -y
ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]

#mysql+mysqldb://raj:admin@localhost:3306/airflow_db