from train_helpers.train_resnet18 import ResNet18Trainer
from train_helpers.train_resnet50 import ResNet50Trainer
from train_helpers.train_resnet101 import ResNet101Trainer
from train_helpers.train_googlenet import GoogleNetTrainer
from train_helpers.train_mobilenetv2 import MobileNetV2Trainer
from train_helpers.train_inceptionv3 import InceptionV3Trainer
from train_helpers.train_xception import XceptionTrainer
from train_helpers.train_squeezenet import SqueezeNetTrainer
import os
import yaml
import time

# ==========================================================
# __________________ IMPORTANT NOTE ________________________

# UPDATE THE data_ID and the RUN NUMBER BEFORE RUNNING THE SCRIPT

# ==========================================================

run = 1
data_ID = "012345_ID"

# Define the models to be used
models = [
    # ResNet18Trainer,
    # ResNet50Trainer, 
    # ResNet101Trainer, 
    GoogleNetTrainer,
    # MobileNetV2Trainer,
    # InceptionV3Trainer, 
    # XceptionTrainer, 
    # SqueezeNetTrainer, 
]

# define path to main config file
config_file = 'models_config_base/training_config.yaml'

# get the number of classes from the folder names inside the train folder
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    config['num_classes'] = len(os.listdir(config['train_data_path']))
# save the updated config file
with open(config_file, 'w') as f:
    yaml.dump(config, f)

# take start time
start_time = time.time()
# Loop over the models and train each model
for Model in models:
    print('Training model: {}'.format(Model.__name__))

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update the config file
    config['num_epochs'] = 1
    # update the model load path 
    config['load_model_path'] = config['load_model_path']+Model.__name__[:-7]+'.pth'
    # update the model save path
   
    config['save_model_path'] = config['save_model_path']+Model.__name__[:-7]+'-' + data_ID + '-run-' + str(run) + '.pth'

    # save to a new folder: models_config_saved
    base_name = 'models_config_saved/' + Model.__name__[:-7]

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    new_config_file = "{}-{}-run-{}{}".format(base_name, data_ID, run, '.yaml')
   
    # Save the updated config file
    with open(new_config_file, 'w') as f:
        yaml.dump(config, f)

    # Load the updated config file
    with open(new_config_file, 'r') as f:
        config = yaml.safe_load(f)

    load_model_path = config['load_model_path']
    save_model_path = config['save_model_path']
    train_data_path = config['train_data_path']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    imageResizeSize = config['imageResizeSize']
    imageCropCentre = config['imageCropCentre']
    lr = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    num_classes = config['num_classes']
    validation_split = config['validation_split']
    project = config['project']
    team = config['team']
    patience = config['patience']

    # Create the Model object and train the model
    Model = Model(load_model_path, save_model_path, train_data_path, 
                  batch_size, num_epochs, lr, momentum, weight_decay, 
                  num_classes, validation_split, data_ID, run, project, 
                  team, imageResizeSize, imageCropCentre, patience)
    Model.train()

# take end time
end_time = time.time()
# print the total time taken, show the time in HH:MM:SS format
print('Total time taken: {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))))



