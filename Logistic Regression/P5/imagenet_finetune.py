import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import logging
import copy

NUM_EPOCH = 5
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename='cifar10_resnet.log',level=logging.DEBUG)


class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

def train():
    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    validateset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)
    validateloader = torch.utils.data.DataLoader(validateset, batch_size=4,
                                              shuffle=True, num_workers=2)

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)
    model.to(device)
    bestValidLoss = 1000
    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        logging.info('Epoch No: {}'.format(epoch))
        running_loss = 0.0
        training_loss = 0.0
        model.train(True) #Training Phase
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        training_loss /= len(trainset)
        logging.info('Training Loss: {:.4f}'.format(training_loss))
        validation_loss = 0.0

        model.train(False) #Validation Phase
        for i, data in enumerate(validateloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            #if i % 20==19:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
        validation_loss /= len(validateset)
        if(validation_loss < bestValidLoss):
            bestValidLoss = validation_loss
            bestModel = copy.deepcopy(model.state_dict())
        logging.info('Validation Loss: {:.4f}'.format(validation_loss))
    filename = 'finalized_tck_model.sav'
    pickle.dump(bestModel, open(filename, 'wb'))
    print('Finished Training')


if __name__ == '__main__':
    train()