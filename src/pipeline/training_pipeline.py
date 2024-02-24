from src.components.stage_7_model_test_data_prediction import model_tester_component
# import os


class TrainingPipeline(model_tester_component):
    def __init__(self):
        super().__init__()

    def data_ingestion_(self):
        self.data_ingestion()

    def initial_processing_(self):
        self.initial_processing()

    def data_validation__(self):
        self.data_validation_()

    def data_splitting_(self):
        self.data_splitting()

    def final_processing_(self):
        self.final_processing()

    def models_tuning_(self):
        self.models_tuning()

    def model_testing_(self):
        self.model_testing()


# training_pipeline_obj = TrainingPipeline()
# training_pipeline_obj.data_ingestion_()
# training_pipeline_obj.initial_processing_()
# training_pipeline_obj.data_validation__()
# training_pipeline_obj.data_splitting_()
# training_pipeline_obj.final_processing_()
# training_pipeline_obj.models_tuning_()
# training_pipeline_obj.model_testing_()
