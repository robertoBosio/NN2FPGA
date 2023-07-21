from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from tqdm import tqdm
from utils import ImageNetKaggle
model = torchvision.models.resnet50(weights="DEFAULT")
model.eval().cuda()  # Needs CUDA, don't bother on CPUs
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
path_to_dataset = "/home/teodoro/datasets/Kaggle_imagenet/"
dataset = ImageNetKaggle(path_to_dataset, "val", val_transform)
dataloader = DataLoader(
            dataset,
            batch_size=64, # may need to reduce this depending on your GPU 
            num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
correct = 0
total = 0
with torch.no_grad():
    for x, y in tqdm(dataloader):
        y_pred = model(x.cuda())
        correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
        total += len(y)
print(correct / total)