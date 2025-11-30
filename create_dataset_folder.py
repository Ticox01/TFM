import os
import random
import shutil

dataset_path = "Datasets/FADC_DATASET"
dataset_autistic_folder = "ASD_children"
dataset_nonautistic_folder = "TD"
autistic_folder = "autistic"
nonautistic_folder = "non_autistic"
folders_to_create = ["test","train","valid"]
max_images = 500
train_porcentaje = 80
valid_porcentaje = 20

def genearte_train_valid_test(images_list, minmum_len, train_porcentaje, valid_porcentaje, max_images):

    images_train_number = int(minmum_len * train_porcentaje / 100)
    images_test_number = minmum_len - images_train_number
    images_valid_number = int(images_train_number * valid_porcentaje / 100)
    images_train_number = images_train_number - images_valid_number

    children = list(images_list.keys())
    random.shuffle(children)

    images_train_count = 0
    images_valid_count = 0
    images_test_count = 0

    train = []
    valid = []
    test = []
    for child in children:
        if child == "length":
            continue

        if len(images_list[child]) >= max_images:
            images = random.sample(images_list[child], 50)
        else:
            images = images_list[child]

        if images_train_count < images_train_number:
            train.extend(images)
            images_train_count = images_train_count + len(images)
        elif images_valid_count < images_valid_number:
            valid.extend(images)
            images_valid_count = images_valid_count + len(images)
        elif images_test_count < images_test_number:
            test.extend(images)
            images_test_count = images_test_count + len(images)
        else:
            break
    
    print("Valores calculados:")
    print(images_train_number,images_valid_number,images_test_number)
    print("Valores Obtenidos:")
    print(len(train), len(valid), len(test))

    return train, valid, test

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

autistic_images = {}
non_autistic_images = {}

# Create dict with autistic children
length = 0
for child in os.listdir(f"{dataset_path}/{dataset_autistic_folder}"):
    for image in os.listdir(f"{dataset_path}/{dataset_autistic_folder}/{child}"):
        if child not in autistic_images.keys():
            autistic_images[child] = [f"{dataset_path}/{dataset_autistic_folder}/{child}/{image}"]
        else:
            autistic_images[child].append(f"{dataset_path}/{dataset_autistic_folder}/{child}/{image}")
        length = length + 1

autistic_images["length"] = length
# Create dict with non autistic children
length = 0
for image in os.listdir(f"{dataset_path}/{dataset_nonautistic_folder}"):
    child = image.split(".")[0]
    if child not in non_autistic_images.keys():
        non_autistic_images[child] = [f"{dataset_path}/{dataset_nonautistic_folder}/{image}"]
    else:
        non_autistic_images[child].append(f"{dataset_path}/{dataset_nonautistic_folder}/{image}")
    length = length + 1

non_autistic_images["length"] = length

minmum_len = min(non_autistic_images["length"], autistic_images["length"])

train_images_autisitc, valid_images_autisitc, test_images_autisitc = genearte_train_valid_test(autistic_images, minmum_len, train_porcentaje, valid_porcentaje, max_images)
train_images_nonautisitc, valid_images_nonautisitc, test_images_nonautisitc = genearte_train_valid_test(non_autistic_images, minmum_len, train_porcentaje, valid_porcentaje, max_images)

train_path = f"{dataset_path}/train"
valid_path = f"{dataset_path}/valid"
test_path = f"{dataset_path}/test"
generate_datasets_files(train_images_autisitc, train_images_nonautisitc, train_path)
generate_datasets_files(valid_images_autisitc, valid_images_nonautisitc, valid_path)
generate_datasets_files(test_images_autisitc, test_images_nonautisitc, test_path)