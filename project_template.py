import os
import re  # type: ignore

#####################################################################
# To create folders
folders = [
    'src',
    os.path.join('src', 'components'),
    os.path.join('src', 'pipeline'),
    os.path.join('src', 'config'),
    os.path.join('src', 'entity'),
    os.path.join('src', 'constants'),

    'artifacts',
    os.path.join('artifacts', 'data'),
    os.path.join('artifacts/data', 'raw'),
    os.path.join('artifacts/data', 'temp'),
    os.path.join('artifacts/data', 'processed'),
    os.path.join('artifacts/data/processed', 'stage_1_initial_processing'),
    os.path.join('artifacts/data/processed', 'stage_2_validation'),
    os.path.join('artifacts/data/processed', 'stage_3_final_processing'),
    os.path.join('artifacts/data', 'train_test'),
    os.path.join('artifacts/data', 'final_testing_data_and_predicted_data'),

    os.path.join('artifacts', 'model'),
    os.path.join('artifacts/model', 'hp_tuned_model'),

    os.path.join('artifacts', 'metrics'),
    os.path.join('artifacts', 'preprocessor'),

    'Secrets',
    os.path.join('Secrets', 'Bundles'),
    os.path.join('Secrets', 'Keys'),
    os.path.join('Secrets', 'Tokens'),
    os.path.join('Secrets', 'Secrets'),

    'templates',
    'static',
    'config'
]
for i in folders:
    os.makedirs(i, exist_ok=True)
    if re.findall('src', i):
        with open(os.path.join(i, '__init__.py'), 'w'):
            pass

#####################################################################
# To create files
files = [
    os.path.join('src', 'utils.py'),
    'schema.yaml',
    'params.yaml',
    'dvc.yaml'
    'app.py',
    'setup.py',
    os.path.join('config', 'config.yaml'),
    os.path.join('templates', 'index.html'),
    os.path.join('templates', 'result.html')
]
for i in files:
    with open(i, 'w'):
        pass

#####################################################################
# To create .gitkeep file in empty folders for initial git committing

for i in folders:
    if any(os.listdir(i)):
        pass
    else:
        with open(os.path.join(i, '.gitkeep'), 'w'):
            pass
