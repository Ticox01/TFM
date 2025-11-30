import os
import shutil
    

dataset_path = "Datasets/FADC_DATASET/ASD_children"
child_number = 0

for child in os.listdir(dataset_path):
    child_image = 0
    for image in os.listdir(f"{dataset_path}/{child}"):
        shutil.copyfile(f"{dataset_path}/{child}/{image}", f"{dataset_path}/{child}/{child}_{child_image}.jpg")
        os.remove(f"{dataset_path}/{child}/{image}")
        child_image = child_image + 1
    child_number = child_number + 1

dataset_path = "Datasets/FADC_DATASET/TD"


for child in os.listdir(dataset_path):
    shutil.copyfile(f"{dataset_path}/{child}", f"{dataset_path}/test/child_{child_number}.jpg")
    os.remove(f"{dataset_path}/{child}")
    child_number = child_number + 1