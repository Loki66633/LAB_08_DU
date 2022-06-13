import PIL
import numpy as np
import pkbar as pkbar
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
from torchsummary import summary
from model import *


def showDataset(dataset):
    class_names = dataset.classes
    picAsArrayArrays = dataset.data
    print(picAsArrayArrays)
    plt.figure(figsize=(10, 10))
    print(picAsArrayArrays[0])
    plt.suptitle("CIFAR-10")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Image.fromarray(picAsArrayArrays[i]))
        plt.xlabel(class_names[dataset.targets[i]])
    plt.show()





cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')
torch.cuda.empty_cache()
print(cuda)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAutocontrast(0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10('cifar_data', download=True, train=True, transform=transform_train)
testset = datasets.CIFAR10('cifar_data', download=True, train=False, transform=transform_test)

batch_size = 256
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

print("Total No of Images in dataset:", len(trainset) + len(testset))
print("No of images in Training dataset:    ", len(trainset))
print("No of images in Testing dataset:     ", len(testset))

l = trainset.classes
l.sort()
print("No of classes: ", len(l))
print("List of all classes")
print(l)

#showDataset(trainset)
model = ResNet75(img_channels=3, num_classes=10)
print(model)
if torch.cuda.is_available():
    model.cuda()

#summary(model.cuda(), (3, 32, 32))

loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adadelta(model.parameters())

#59 64 DROP
#63 50 BEZ DROP
#52 59 DROP(0.1) +
#66 61


epochs = 16
epoch = 0
checkpoint = torch.load('//home/pc/PycharmProjects/LAB_08/checkpoints/chk_2_75.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch'] + 1
loss = checkpoint['loss']
train_per_epoch = int(len(trainset) / batch_size)
for e in range(epoch, epochs):
    print("")
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=e, num_epochs=epochs, width=20, always_stateful=False)
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        output = model(images)

        labels = labels.to(device)

        loss = loss_fn(output, labels)

        loss.backward()

        optimizer.step()



        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == labels).sum().item()
        accuracy = correct / len(predictions)
        kbar.update(idx, values=[("loss", loss), ("acc", accuracy)])

    if e % 5 == 0:
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn
        }, 'checkpoints/chk_2_' + str(e) + '.pth')

num_correct = 0
num_samples = 0
model.eval()
print("\n")
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(
        f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')
























"""
with torch.no_grad():

    image = Image.open("/home/pc/PycharmProjects/LAB_06_DU/frog.jpg")
    original = Image.open("/home/pc/PycharmProjects/LAB_06_DU/frog.jpg")
    image = image.resize((32,32))
    image = tt.ToTensor()(image)
    image = image.view(1,3,32,32)

    # Generate prediction
    prediction = model(image.to(device))

    # Predicted class value using argmax
    predicted_class = np.argmax(prediction.cpu())

    # Show result
    predicted_class = trainset.classes[(predicted_class)]

    plt.clf()
    plt.imshow(original)
    plt.title(f'Prediction: {predicted_class}')
    plt.show()
"""
