import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os

# Define the class labels
class_labels = ['Tiger', 'Lion', 'Cheetah', 'Leopard', 'Puma']

# Define the custom dataset class
class FelidaeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = self.get_file_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.get_label_from_filename(img_path)
        return image, label

    def get_file_list(self):
        file_list = []
        for class_label in class_labels:
            class_dir = os.path.join(self.root_dir, class_label)
            files = os.listdir(class_dir)
            files = [os.path.join(class_dir, file) for file in files]
            file_list.extend(files)
        return file_list

    def get_label_from_filename(self, filename):
        for i, class_label in enumerate(class_labels):
            if class_label.lower() in filename.lower():
                return i
        return -1

# Define the transformations for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(299),  # InceptionV3 expects 299x299 sized images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = models.inception_v3(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_labels))
model = model.to(device)
model.eval()

# Load the test dataset
test_dataset = FelidaeDataset(root_dir='./images', transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model on the test dataset
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())
count_cheetah = 0
count_leopard = 0
count_lion = 0
count_puma = 0
count_tiger = 0
correct_cheetah = 0
correct_leopard = 0
correct_lion = 0
correct_puma = 0
correct_tiger = 0
# Print the predictions
for i, prediction in enumerate(predictions):
    img_path = test_dataset.file_list[i]
    img_name = os.path.basename(img_path)
    convert_img_name = img_name.split("_")[0]
    count_cheetah += convert_img_name == "Cheetah"
    count_leopard += convert_img_name == "Leopard"
    count_lion += convert_img_name == "Lion"
    count_puma += convert_img_name == "Puma"
    count_tiger += convert_img_name == "Tiger"
    
    correct_cheetah += convert_img_name == "Cheetah" and class_labels[prediction] == "Cheetah"
    correct_leopard += convert_img_name == "Leopard" and class_labels[prediction] == "Leopard"
    correct_lion += convert_img_name == "Lion" and class_labels[prediction] == "Lion"
    correct_puma += convert_img_name == "Puma" and class_labels[prediction] == "Puma"
    correct_tiger += convert_img_name == "Tiger" and class_labels[prediction] == "Tiger"
    
    print(f"Image: {img_name}, Predicted Class: {class_labels[prediction]}")
     
print("Correct Cheetah : ", correct_cheetah, "Count Cheetah : ", count_cheetah, "Accuracy Cheetah : ",  correct_cheetah / count_cheetah * 100, "%")
print("Correct Leopard : ", correct_leopard, "Count Leopard : ", count_leopard, "Accuracy Leopard : ",  correct_leopard / count_leopard * 100, "%")
print("Correct Lion : ", correct_lion, "Count Lion : ", count_lion, "Accuracy Lion : ",  correct_lion / count_lion * 100, "%")
print("Correct Puma : ", correct_puma, "Count Puma : ", count_puma, "Accuracy Puma : ",  correct_puma / count_puma * 100, "%")
print("Correct Tiger : ", correct_tiger, "Count Tiger : ", count_tiger, "Accuracy Tiger : ",  correct_tiger / count_tiger * 100, "%")
