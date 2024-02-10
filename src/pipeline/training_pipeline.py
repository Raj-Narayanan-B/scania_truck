from src.components.stage_6_model_test_data_prediction import model_trainer_component


class Training_Pipeline(model_trainer_component):
    def __init__(self):
        super().__init__()

    def data_ingestion_(self):
        self.data_ingestion()

    def initial_processing_(self):
        self.initial_processing()

    def data_splitting_(self):
        self.data_splitting()

    def final_processing_(self):
        self.final_processing()

    def models_tuning_(self):
        self.models_tuning()

    def model_training_(self):
        self.model_training()


training_pipeline_obj = Training_Pipeline()
training_pipeline_obj.data_ingestion_()
training_pipeline_obj.initial_processing_()
training_pipeline_obj.data_splitting_()
training_pipeline_obj.final_processing_()
training_pipeline_obj.models_tuning_()
training_pipeline_obj.model_training_()
