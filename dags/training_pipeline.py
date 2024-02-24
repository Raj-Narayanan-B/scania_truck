#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# from __future__ import annotations

# [START tutorial]
# [START import_module]
import textwrap  # type: ignore

import pendulum

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

from src.pipeline.training_pipeline import TrainingPipeline
training_pipeline_obj = TrainingPipeline()

# [END import_module]

# [START instantiate_dag]
with DAG(
    "training_pipeline",
    # [START default_args]
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={"retries": 2},
    # [END default_args]
    description="This pipeline will generate the best model out of 14 models for the data that is input to this pipeline",
    schedule="@monthly",
    start_date=pendulum.datetime(2024, 2, 20, tz="UTC"),
    catchup=False,
    tags=["machine_learning", "classification", "sensor"],
    access_control={
        "Admin": {"can_edit", "can_read", "can_delete"},
    }
) as dag:
    # [END instantiate_dag]
    # [START documentation]
    dag.doc_md = __doc__
    # [END documentation]

    # [START data_ingestion]
    def data_ingestion(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.data_ingestion_()
        # ti.xcom_push("order_data", data_string)

    def initial_processing(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.initial_processing_()

    # def initial_processing(**kwargs):
    #     ti = kwargs["ti"]
    #     extract_data_string = ti.xcom_pull(task_ids="extract", key="order_data")
    #     order_data = json.loads(extract_data_string)

    #     total_order_value = 0
    #     for value in order_data.values():
    #         total_order_value += value

    #     total_value = {"total_order_value": total_order_value}
    #     total_value_json_string = json.dumps(total_value)
    #     ti.xcom_push("total_order_value", total_value_json_string)
    def data_validation(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.data_validation__()

    def data_splitting(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.data_splitting_()

    def final_processing(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.final_processing_()

    def models_tuning(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.models_tuning_()

    def model_testing(**kwargs):
        ti = kwargs["ti"]  # noqa
        training_pipeline_obj.model_testing_()

    # def load(**kwargs):
    #     ti = kwargs["ti"]
    #     total_value_string = ti.xcom_pull(task_ids="transform", key="total_order_value")
    #     total_order_value = json.loads(total_value_string)

    #     print(total_order_value)

    # [START main_flow]
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = textwrap.dedent(
        """\
    #### Ingestion task
    This task extracts the data from DB in batches. Both training dataset and testing dataset are downloaded in batches
    (3 for each type of dataset). Totally, there will be 6 batches of data retrieved from the DB.
    All these 6 raw data batches are tracked and versioned by DVC.
    """
    )

    initial_processing_task = PythonOperator(
        task_id="initial_processing",
        python_callable=initial_processing,
    )
    initial_processing_task.doc_md = textwrap.dedent(
        """\
    #### Initial Processing task
    This task merges the batch data and does basic pre-processing
    (such as renaming target name to 'class', sorting the rows/entries by ascending order of
    PRIMARY KEY and dropping the PRIMARY KEY column that was received from DB).
    This task also does initial/basic validation (such as mapping target values from
    neg->0 and pos->1, converting 'na' to np.nan, converting the dtypes of input features to float).
    This is done on both - Training and Testing data.
    The preprocessed Training and Testing datasets are tracked and versioned by DVC.
    """
    )

    data_validation_task = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation,
    )
    data_validation_task.doc_md = textwrap.dedent(
        """\
    #### Data Validation task
    This advanced data validation is done only on the training dataset.
    This task does advanced validation such as: removing features that have more than 50%
    missing values (removed features are saved in schema), removing featuers that have 0 standard deviation
    (removed features are saved in schema), identifying histogram features if any.
    The validated data is tracked and versioned by DVC.
    """
    )

    data_splitting_task = PythonOperator(
        task_id="data_splitting",
        python_callable=data_splitting,
    )
    data_splitting_task.doc_md = textwrap.dedent(
        """\
    #### Data Splitting task
    This task splits the validated training data into training data subset and validation data subset.
    The training data subset and validation data subset are tracked and versioned by DVC.
    """
    )

    final_processing_task = PythonOperator(
        task_id="final_processing",
        python_callable=final_processing,
    )
    final_processing_task.doc_md = textwrap.dedent(
        """\
    #### Final Processing task
    In this final processing task, the missing value imputation, scaling of data, data imbalance addressing and
    dropping of duplicate values is done. The final schema of the required columns is saved. The preprocessor object
    containing the imputer and scaler is saved. This is done only on the training data subset.
    The validation data subset is transformed in accordance with the pipeline saved earlier.
    This validation data subset is also balanced w.r.t the data_imbalance problem.

    Finally, the duplicates(if any) are dropped in validation data subset as well.
    The transformed training data subset and validation data subset are tracked and versioned by DVC.
    """
    )

    models_tuning_task = PythonOperator(
        task_id="models_tuning",
        python_callable=models_tuning,
    )
    models_tuning_task.doc_md = textwrap.dedent(
        """\
    #### Models Tuning task
    This task does hyper-parameter tuning on the entire training data subset by splitting it into 12 batches. The tool used
    for hyperparameter tuning is Optuna. Overall 3 trials are alloted for each model. For each trial, a set of hyperparameter set
    is chosen for a model. Around 12 to 14 different machine learning models are tried on the batches
    with different sets of hyperparameters.
    In a trial, if a selected hyperparameter set for a model yields
    unpromising trials on subsequent batches, that trial will be pruned to save resources and time. The next trial will commence
    if any trial is left or if not, then the control will move on to the next available model.
    In this extensive hyperparamter tuning, the artifacts produced (the models, the selected parameters and the metrics for every trial,
    and for every batches) is tracked by MLflow. At the end of hyperperameter tuning, the best model that did not get pruned
    for any of the batches in a trial will be chosen as the best Hyper-Parameter tuned candidate. Then the next 6 best models are chosen
    and given as estimators for stacking classifier and voting classifer with the best hyperparameter tuned candidate being the
    final estimator in the stacking classifier.

    Finally, there will be 3 challenger models (1st - Best Hyperparameter Tuned Candidate, 2nd - Stacking Classifer, 3rd - Voting
    Classifier). These challenger models will now trained on the entire training data subset and will be tracked and versioned
    by MLflow.

    The best hyperparameter tuned candidate, stacking classifier, voting classifier, the dataframe
    from Optuna's HP trials are tracked and versioned by DVC.
    With this, the task of Models-Tuning concludes.
    """
    )

    model_testing_task = PythonOperator(
        task_id="model_testing",
        python_callable=model_testing,
    )
    model_testing_task.doc_md = textwrap.dedent(
        """\
    #### Model Testing task
    In this task, advanced Data validation is applied on the testing dataset (NOTE: basic data validation for this testing data
    is already done during the Initial Processing task). This task retrieves the 3 champion models from MLflow saved from previous
    task, and all three models are used to predict the testing_data. Whichever model gives the highest accuracy will
    be chosen as the champion model.
    The champion is also tracked by DVC and is logged with alias name: Champion in MLflow.
    The validated test data, the predicted data are tracked and versioned by DVC.
    """
    )

    data_ingestion_task >> initial_processing_task >> data_validation_task >> data_splitting_task >> final_processing_task >> models_tuning_task >> model_testing_task

# [END main_flow]

# [END tutorial]
