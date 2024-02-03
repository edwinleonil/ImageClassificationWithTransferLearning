import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import equalize
import numpy as np
import pandas as pd

class Resnet50Tester:
    def __init__(self, model_path, test_dir):
        self.model_path = model_path
        self.test_dir = test_dir
        self.num_classes = 9
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: equalize(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    def test(self):
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
        self.probs["ground_truth"] = self.ground_truth
        self.probs["predictions"] = self.preds
        self.probs["predictions"] = self.probs["predictions"].apply(lambda x: self.classes[x])
        self.probs["ground_truth"] = self.probs["ground_truth"].apply(lambda x: self.classes[x])
        cols = self.probs.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        self.probs = self.probs[cols]
        self.probs.to_csv("test_results/probs_resnet50.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    resnet50_tester = Resnet50Tester("trained_models/resnet50.pth", "data/test")
    resnet50_tester.test()

    