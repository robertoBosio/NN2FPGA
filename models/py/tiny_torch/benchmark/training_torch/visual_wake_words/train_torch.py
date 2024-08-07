import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from vww_torch import MobileNetV1
from tqdm import tqdm

# IMAGE_SIZE = 96
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
BASE_DIR = os.path.join("/home-ssd/datasets/vw", 'vw_coco2014_96')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) >= 2:
        model = MobileNetV1(num_filters=3, num_classes=2)
        model.load_state_dict(torch.load(sys.argv[1]))
    else:
        model = MobileNetV1(num_filters=3, num_classes=2)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=BASE_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = torchvision.datasets.ImageFolder(root=BASE_DIR, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(train_dataset.class_to_idx)

    model = train_epochs(model, device, train_loader, val_loader, EPOCHS, 0.01, criterion)
    model = train_epochs(model, device, train_loader, val_loader, 10, 0.0005, criterion)
    model = train_epochs(model, device, train_loader, val_loader, 20, 0.00025, criterion)

    # Save model
    if len(sys.argv) >= 3:
        torch.save(model.state_dict(), sys.argv[2])
    else:
        torch.save(model.state_dict(), 'trained_models/vww_96.pt')

def train_epochs(model, device, train_loader, val_loader, epoch_count, learning_rate, criterion):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_count):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        # Write tqdm progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                # remove dimensions at 1
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                tepoch.set_postfix({"Acc": f"{100.0 * train_correct / total:.2f}%", "Loss": f"{loss.item():.4f}"})
                train_loss += loss.item()

            train_accuracy = 100.0 * train_correct / total
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            total = 0

            with torch.no_grad():
                with tqdm(val_loader, unit="batch") as tepoch:
                    for inputs, labels in tepoch:
                        tepoch.set_description(f"Epoch {epoch} (Val)")

                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)
                        outputs = outputs.squeeze()
                        loss = criterion(outputs, labels)

                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        val_loss += loss.item()
                        tepoch.set_postfix({"Acc": f"{100.0 * val_correct / total:.2f}%", "Loss": f"{loss.item():.4f}"})

            val_accuracy = 100.0 * val_correct / total
            val_loss /= len(val_loader)

            # print(f"Epoch: {epoch+1}/{epoch_count}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return model

if __name__ == '__main__':
    main()