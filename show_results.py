# plot the confusion matrix with the class names
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import pandas as pd

file_path = r"C:\test_results\Xception-run-1.csv"
df = pd.read_csv(file_path)

ground_truth = df["GroundTruth"]
predictions = df["ModelPrediction"]

# Calculate the confusion matrix
cm = confusion_matrix(ground_truth, predictions)

# compute the accuracy
accuracy = (sum(cm.diagonal()) / sum(sum(cm)))*100

# plot the confusion matrix with the class names
plt.figure(figsize=(15, 10))
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")


# add the accuracy per each class on the right side of the confusion matrix as the heatmap is not clear
# get the list of unique classes
classes = df["GroundTruth"].unique()
# get the accuracy per each class
class_accuracy = []
for class_name in classes:
    # get the total number of images for each class
    total = df[df["GroundTruth"] == class_name].shape[0]
    # get the number of correct predictions for each class
    correct = df[(df["GroundTruth"] == class_name) & (df["GroundTruth"] == df["ModelPrediction"])].shape[0]
    # calculate the accuracy
    class_accuracy.append(correct/total*100)

# add the accuracy per each class on the matrix
for i in range(len(classes)):
    # locate the text in the right of the matrix
    ax.text(-1, i+0.5, f"{class_accuracy[i]:.2f}%", ha="left", va="center", color="black", fontsize=14)

# add labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')

# add model name on the title, get this from the file path
model_name = file_path.split("\\")[-1].split("-")[0]
ax.set_title(f'Confusion Matrix for {model_name}\nAccuracy: {accuracy:.2f}%')

ax.xaxis.set_ticklabels(df["ModelPrediction"].unique())
ax.yaxis.set_ticklabels(df["GroundTruth"].unique())
plt.show()
