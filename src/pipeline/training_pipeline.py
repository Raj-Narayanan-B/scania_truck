from src.components.stage_6_model_test_data_prediction import model_trainer_component


class Training_Pipeline(model_trainer_component):
    def __init__(self):
        super().__init__()

    def training_pipeline(self):
        self.data_ingestion()

        self.initial_processing()

        self.data_splitting()

        self.final_processing()

        self.models_tuning()

        self.model_training()


training_pipeline_obj = Training_Pipeline()
training_pipeline_obj.training_pipeline()
