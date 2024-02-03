import torch
# import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import equalize
import numpy as np
import pandas as pd
import os
import yaml

class Classifier:
    def __init__(self, model_path, test_dir):
        self.model_path = model_path
        self.test_dir = test_dir
        # self.num_classes = 9

        # get model name from the model path
        self.model_name = self.model_path.split("/")[-1].split("-")[0]

        # Load the master training configurations file
        with open('models_config_base/master_training_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Get the configuration for each model
        model_config = config[self.model_name]

        # Apply transforms and augmentations to the images
        self.transform = v2.Compose([
            v2.Resize((model_config['resize'])),
            v2.Grayscale(num_output_channels=1),  # convert image to grayscale
            v2.Lambda(lambda img: equalize(img)),  # equalize image
            v2.ToTensor(),
            v2.Lambda(lambda img: img.repeat(3, 1, 1)),  # duplicate grayscale channel
            v2.Normalize(mean=config['General_config']['mean'], std=config['General_config']['std'])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = torch.load(self.model_path)
        self.resnet50.eval()
        self.test_dataset = ImageFolder(root=self.test_dir, transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        self.class_to_idx = self.test_dataset.class_to_idx
        self.classes = [name for name in self.class_to_idx.keys()]
        self.ground_truth = []
        self.probs = []
        self.preds = []

    def predict(self):
        print(f"Using device: {self.device}")
        for _, labels in self.test_loader:
            self.ground_truth.extend(labels.numpy())
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.resnet50(inputs)
                self.probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
                self.preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        self.probs = np.round(self.probs, 2)
        self.probs = pd.DataFrame(self.probs, columns=self.classes)

        self.probs["Image_ID"] = self.test_dataset.samples
        # only keep the file name
        self.probs["Image_ID"] = self.probs["Image_ID"].apply(lambda x: x[0].split("/")[-1])
        # the output is: test\HAB\Part 1_A00760121_Res_1_Normals X.png
        # only keep the file name which is after the last \
        self.probs["Image_ID"] = self.probs["Image_ID"].apply(lambda x: x.split("\\")[-1])
        self.probs["GroundTruth"] = self.ground_truth
        self.probs["ModelPrediction"] = self.preds
        self.probs["GroundTruth"] = self.probs["GroundTruth"].apply(lambda x: self.classes[x])
        self.probs["ModelPrediction"] = self.probs["ModelPrediction"].apply(lambda x: self.classes[x])
        # place the GroundTruth and ModelPrediction columns to the first two columns
        cols = self.probs.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        self.probs = self.probs[cols]
        # place the Image_ID column to the first column
        cols = self.probs.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.probs = self.probs[cols]
        # save using the model name
        model_name = self.model_path.split("/")[-1].split(".")[0]

        # save the results to a csv file
        self.probs.to_csv(f"test_results/{model_name}.csv", index=False, float_format="%.2f")

if __name__ == "__main__":

    # ==================================================================
    # ______________________ UPDATE THESE VALUES ______________________
    # define the run number and the dataset ID before running the script

    run_number = 1
    dataset_ID = "9classes"

    # ______________________ UPDATE THESE VALUES ______________________
    # ==================================================================

    # get the list of trained models
    trained_models = os.listdir("trained_models")

    # define the path to the test data
    test_data_path = f"data/{dataset_ID}/test"
    
    # condition for selecting the trained models
    trained_models = [model for model in trained_models if f"run-{run_number}" in model and dataset_ID in model]

    # if list is empty, print a message
    if len(trained_models) == 0:
        print(f"No trained models with run number {run_number} found")
    else:
        for model in trained_models:
            print(model)
        for model in trained_models:
            print(f"Predicting using {model}")
            defect_classifier = Classifier(f"trained_models/{model}", test_data_path)
            defect_classifier.predict()
            print(f"Done predicting using {model}")

    