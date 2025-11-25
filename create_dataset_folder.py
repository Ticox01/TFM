import os
import random
import shutil

dataset_path = "Datasets/FADC_DATASET"
dataset_autistic_folder = "ASD"
dataset_nonautistic_folder = "TD"
autistic_folder = "autistic"
nonautistic_folder = "non_autistic"
folders_to_create = ["test","train","valid"]
train_porcentaje = 86
valid_porcentaje = 26

def genearte_train_valid_test(images_list, train_porcentaje, valid_porcentaje):
    images_train_number = int(len(images_list) * train_porcentaje / 100)
    images_valid_number = int(images_train_number * valid_porcentaje / 100)

    all_train_images = random.sample(images_list, images_train_number)
    valid_images = random.sample(all_train_images, images_valid_number)
    train_images = [x for x in all_train_images if x not in valid_images]
    test_images = [x for x in images_list if x not in all_train_images]

    return train_images, valid_images, test_images

def copy_image_to_final_folder(image_path, final_folder, index):
    shutil.copyfile(image_path, final_folder+f"/{index:04d}.jpg")

def generate_datasets_files(images_autisitc, non_autisitc, path):
    for idx, file in enumerate(images_autisitc):
        copy_image_to_final_folder(file, path + "/" + autistic_folder, idx)
    for idx, file in enumerate(non_autisitc):
        copy_image_to_final_folder(file, path + "/" + nonautistic_folder, idx)

for folder in folders_to_create:
    if not os.path.exists(dataset_path+"/"+folder):
        os.makedirs(dataset_path+"/"+folder+"/"+autistic_folder)
        os.makedirs(dataset_path+"/"+folder+"/"+nonautistic_folder)

autistic_images = []
non_autistic_images = []

for folder in os.listdir(dataset_path):
    for image in os.listdir(dataset_path+"/"+folder):
        if folder == dataset_autistic_folder:
            autistic_images.append(dataset_path+"/"+folder+"/"+image)
        elif folder == dataset_nonautistic_folder:
            non_autistic_images.append(dataset_path+"/"+folder+"/"+image)

minmum_len = min(len(autistic_images), len(non_autistic_images))
random.shuffle(autistic_images)
random.shuffle(non_autistic_images)
autistic_images = autistic_images[0:minmum_len]
non_autistic_images = non_autistic_images[0:minmum_len]

train_images_autisitc, valid_images_autisitc, test_images_autisitc = genearte_train_valid_test(autistic_images, train_porcentaje, valid_porcentaje)
train_images_nonautisitc, valid_images_nonautisitc, test_images_nonautisitc = genearte_train_valid_test(non_autistic_images, train_porcentaje, valid_porcentaje)

train_path = f"{dataset_path}/train"
valid_path = f"{dataset_path}/valid"
test_path = f"{dataset_path}/test"
generate_datasets_files(train_images_autisitc, train_images_nonautisitc, train_path)
generate_datasets_files(valid_images_autisitc, valid_images_nonautisitc, valid_path)
generate_datasets_files(test_images_autisitc, test_images_nonautisitc, test_path)