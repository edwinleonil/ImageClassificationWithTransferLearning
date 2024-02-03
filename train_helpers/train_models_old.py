from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import equalize
import wandb
import numpy as np
import yaml

# NOTE: # the xception model need the module timm installed (pip install timm)

class ModelTrainer:
    def __init__(self, load_model_path, save_model_path, train_data_path, 
                 num_classes, data_ID, run, model_name):
        
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.train_data_path = train_data_path
        self.data_ID = data_ID
        self.run = run
        self.num_classes = num_classes
        self.model_name = model_name

        # Load the master training configurations file
        with open('models_config_base/master_training_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Get the configuration for each model
        model_config = config[self.model_name]
        general_config = config['General_config']
        self.batch_size = general_config['batch_size']
        self.num_epochs = general_config['num_epochs']
        self.lr = general_config['lr']
        self.momentum = general_config['momentum']
        self.weight_decay = general_config['weight_decay']
        self.validation_split = general_config['validation_split']
        self.patience = general_config['patience']
        self.team = general_config['team']
        self.project_name = general_config['project']

        self.run_name =  self.model_name + '-' + self.data_ID + '-run-' + str(self.run)


        # Define the transforms for each model using the configuration file
        self.transform = transforms.Compose([
            transforms.Resize(model_config['resize']),
            transforms.CenterCrop(model_config['center_crop']),
            transforms.Lambda(lambda img: equalize(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config['mean'], std=model_config['std'])
        ])


        # Define the configuration for the wandb run for each model
        self.config = {
            "Resize": model_config['resize'],
            "center_crop": model_config['center_crop'],
            "epochs": general_config['num_epochs'], 
            "learning_rate": general_config['lr'], 
            "momentum": general_config['momentum'],
            "weight_decay": general_config['weight_decay'],
            "batch_size": general_config['batch_size'],
            "validation_split": general_config['validation_split'],
            "num_classes": num_classes,
            "patience": general_config['patience'],
            "dataset": data_ID,
        }


        # Load each pre-trained model
        print(" the model path is: ", self.load_model_path)
        self.model = torch.load(self.load_model_path)

        # some models have a fc layer, some have a classifier layer and some have a last_linear layer
        # hence we need to check which model we are using and replace the last layer accordingly
        if self.model_name != "MobileNetV2" and self.model_name != "SqueezeNet" and self.model_name != "ViT_base_patch16_224":
            # Replace the last layer with a new one that has num_classes output features
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if self.model_name == "ViT_base_patch16_224":
            # Replace the last layer with a new one that has num_classes output features
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)

        else:
            # if model is mobileNetV2, replace the last layer with a new one that has num_classes output features
            if self.model_name == "MobileNetV2":
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.model.last_channel, num_classes),
                )

            # if model is squeezenet, replace the last layer with a new one that has num_classes output features
            if self.model_name == "SqueezeNet":
                self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                self.model.num_classes = num_classes

            
        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # NOTE: Adam is well known to perform worse than SGD for image classification tasks (chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://opt-ml.org/papers/2021/paper53.pdf)
        # Move the model to the GPU device if available

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Print the device that is being used
        print(f"Using device: {self.device}")

    
    def load_data(self):
            
            # Load the dataset
            dataset = ImageFolder(root=self.train_data_path, transform=self.transform)
            # Get labels from the dataset
            targets = np.array(dataset.targets)
            # Create a StratifiedShuffleSplit instance
            # NOTE: StratifiedShuffleSplit is used to ensure that the training and validation sets have the same class distribution
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_split, random_state=0)

            # Get indices for training and validation sets
            for train_index, val_index in sss.split(np.zeros(len(targets)), targets):
                train_dataset = torch.utils.data.Subset(dataset, train_index)
                valid_dataset = torch.utils.data.Subset(dataset, val_index)

            # Create data loaders for the training and validation sets
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

    def train_epoch(self, epochs):

        # Training phase
        self.model.train()
        running_loss = 0.0

        # NOTE: use to be: for i, (inputs, labels) in enumerate(self.train_loader):
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            # NOTE: this is a temporary fix so that the inception model works
            if self.model_name == "InceptionV3":
                outputs,_ = self.model(inputs) # Get the logits from the InceptionOutputs tuple
            else:
                outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self):

        # Validation phase
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            
            # NOTE: use to be: for i, (inputs, labels) in enumerate(self.valid_loader):
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

        return valid_loss / len(self.valid_loader)

    def train(self):

        # Initialize the best loss to a very high value
        best_loss = float('inf')
        # Initialize the number of epochs since the loss last improved
        epochs_since_improvement = 0
        # Load the data
        self.load_data()

        # Initialize a new wandb run
        with wandb.init(project=self.project_name, name=self.run_name, 
                        config=self.config, entity=self.team):

            # Train the model with early stopping if the validation loss does not improve for 10 epochs
            for epoch in range(self.num_epochs):
                avg_train_loss = self.train_epoch(epoch)
                avg_valid_loss = self.validate_epoch()
                # Print statistics
                print(f"[Epoch {epoch+1}] Train loss: {avg_train_loss:.4f} | Validation loss: {avg_valid_loss:.4f}")
                # log to wandb
                wandb.log({"Epoch": epoch+1, "Train Loss": avg_train_loss, "Validation Loss": avg_valid_loss})
                # If the average validation loss is lower than the best loss so far, update the best loss and save the model
                if avg_valid_loss < best_loss:
                    best_loss = avg_valid_loss
                    print("Validation loss improved, saving the model...")
                    torch.save(self.model, self.save_model_path)
                    # Reset the number of epochs since the loss last improved
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                # If the validation loss does not improve for the patience number of epochs, stop training
                if epochs_since_improvement == self.patience:
                    print("Early stopping due to no improvement in validation loss for 5 epochs.")
                    # close wandb run
                    wandb.finish()
                    return
                
            print("Training completed successfully.")
            # close wandb run
            wandb.finish()


# Uncomment for testing the trainer with one model
# if __name__ == "__main__":
#     # Define the paths and hyperparameters
#     load_model_path = "pretrained_models/googlenet.pth"
#     save_model_path = "trained_models/googlenet_test.pth"
#     train_data_path = "data/train"
#     num_classes = 9
#     data_ID = "200525"
#     run = 1
#     model_name = "GoogleNet"
#     # Create the trainer object and train the model
#     trainer = ModelTrainer(load_model_path, save_model_path, train_data_path, 
#                                num_classes, data_ID, run, team, model_name)
#     trainer.train()