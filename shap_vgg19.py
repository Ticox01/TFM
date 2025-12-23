import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from collections import Counter
import shap
import numpy as np

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


tensor_transform = v2.Compose([
    v2.ToImage(),  # Convierte PIL → Tensor
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

shap_transform = v2.Compose([
    v2.Lambda(nhwc_to_nchw),
    v2.Normalize(mean=(-1*np.array([0.485, 0.456, 0.406])/np.array([0.229, 0.224, 0.225])).tolist(),
                  std=(1/np.array([0.229, 0.224, 0.225])).tolist()),
    v2.Lambda(nchw_to_nhwc),
])

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    @property
    def classes(self):
        return self.data.classes
    @property
    def get_labels_count(self):
        _, labels = zip(*self.data.imgs)
        return Counter(labels)
    
    def get_image_for_shap(self):
        X = []
        y = []

        for image,label in self.data:
            X.append(image.permute(1, 2, 0).numpy())
            y.append(label)
        
        return np.stack(X),np.stack(y)

def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to("xpu")
    output = modelo(img)
    return output

test_dataset = ImageDataset("Datasets/ASD-FIC_dataset/test/", transform=tensor_transform)
X , y = test_dataset.get_image_for_shap()

to_explain = X[[39,41]]

modelo= models.vgg19(weights="IMAGENET1K_V1")
modelo.classifier[6] = nn.Linear(4096, 2)
modelo = modelo.to("xpu")

modelo.load_state_dict(torch.load("best-model-parameters.pt", weights_only=True))
modelo.eval()

# Check that transformations work correctly
Xtr = torch.Tensor(X)
class_names = ["autistic", "non_autistic"]

topk = 2
batch_size = 10
n_evals = 10000
# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(128,128)", Xtr[0].shape)

# create an explainer with model and image masker
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

# feed only one image
# here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(
    Xtr[1:2],
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:topk],
)

shap_values.data = shap_transform(shap_values.data).cpu().numpy()[0]
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
    true_labels=[class_names[0]],
)