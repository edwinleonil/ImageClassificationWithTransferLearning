
from train_helpers.train_models import ModelTrainer
import os
import yaml
import time

# ==========================================================

# __________________ I M P O R T A N T _____________________

# NOTE: UPDATE THE data_ID and the run number for each run

run = 1
data_ID = "7_classes"
# __________________________________________________________

# ==========================================================


# define path to main config file
config_file = 'models_config_base/master_training_config.yaml'

# get the model names and the number of classes 
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    models = config['models']
    config['General_config']['num_classes'] = len(os.listdir(config['General_config']['train_data_path']))

# save the updated config file
with open(config_file, 'w') as f:
    yaml.dump(config, f)

# take start time
start_time = time.time()
# Loop over the models and train each model
for Model in models:
    print('Training model: {}'.format(Model))

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config[Model]
    general_config = config['General_config']

    # close the config file
    f.close()

    model_config = {
        'load_model_path': general_config['load_model_path']+Model+'.pth',
        'save_model_path': general_config['save_model_path']+Model+'-' + data_ID + '-run-' + str(run) + '.pth',
        'train_data_path': general_config['train_data_path'],
        'batch_size': general_config['batch_size'],
        'num_epochs': general_config['num_epochs'],
        'lr': general_config['lr'],
        'momentum': general_config['momentum'],
        'weight_decay': general_config['weight_decay'],
        'num_classes': general_config['num_classes'],
        'validation_split': general_config['validation_split'],
        'patience': general_config['patience'],
        'model': model_config,
    }

    # save to a new folder: models_config_saved
    base_name = 'models_config_saved/' + Model

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    new_config_file = "{}-{}-run-{}{}".format(base_name, data_ID, run, '.yaml')
   
    # Save the updated config file
    with open(new_config_file, 'w') as f:
        yaml.dump(model_config, f)
    

    # Load the updated config file
    with open(new_config_file, 'r') as f:
        config = yaml.safe_load(f)
   
    # Get the updated values from the config file
    load_model_path = model_config['load_model_path']
    save_model_path = model_config['save_model_path']
    train_data_path = model_config['train_data_path']
    num_classes = model_config['num_classes']  

    # close the config file
    f.close()  

    # Create the Model object and train the model
    ModelTrained = ModelTrainer(load_model_path, save_model_path, train_data_path, 
                                num_classes, data_ID, run, Model)
    ModelTrained.train()

# take end time
end_time = time.time()
# print the total time taken, show the time in HH:MM:SS format
print('Total time taken: {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))))



