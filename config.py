import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, random_split
import torchmetrics
from torch import nn , optim
import os

# define the MLP
class CNN(nn.Module):
  def __init__(self) -> None:
    super(CNN,self).__init__()
    self.cnn=nn.Sequential(nn.Conv2d(1,1,3,1),
                           nn.ReLU(),
                           nn.Conv2d(1,1,3,1),
                           nn.ReLU(),
                           nn.Flatten(),
                           nn.Linear(144,28),
                           nn.ReLU(),
                           nn.Linear(28,10),
                           nn.Softmax(1))
  def forward(self,x):
    return self.cnn(x)

class config:
    name='Activity5'
    epochs=10
    lr=1
    batch_size=32
    shuffle=True
    transform=transforms.ToTensor()
    test_frac=0.2
    dirs = ['datasets', 'models']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = datasets.USPS(root='./datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.USPS(root='./datasets', train=False, download=True, transform=transform)
    _,train_dataset =  random_split(train_dataset, [1-test_frac, test_frac])
    _,test_dataset =  random_split(test_dataset, [1-test_frac, test_frac])
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    model = CNN().to(device)
    metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    model_name='cnn_usps'
    model_file = os.path.join('models', model_name)
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.1)
    wandb_log=False