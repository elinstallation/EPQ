import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
import os

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    model.train()  #training mode
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   #evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                #forwardpass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    #backwards pass optimisee only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def main():
    #load datasets
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet normalization
    ]),
}

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join('datasets', 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join('datasets', 'val'), data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False)
    }

    num_classes = len(image_datasets['train'].classes)

    #load resnet50 pre-trained model and modify it for our task
    model = models.resnet50(pretrained=True)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) 

    model = model.to(device)

    #define loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #train
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

    #save
    torch.save(trained_model.state_dict(), 'resnet50_gender_finetuned.pth')

if __name__ == "__main__":
    main()