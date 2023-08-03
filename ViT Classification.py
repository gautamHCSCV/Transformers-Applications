import os
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pickle
import torchvision
from tqdm.notebook import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

config = dict(
    saved_path="saved_models/vit.pt",
    lr=0.001,
    EPOCHS = 10,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 224,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=2,
    USE_AMP = True,
    channels_last=False)

random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])

torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

my_path = 'classfication final data'
images = torchvision.datasets.ImageFolder(root=my_path,transform=data_transforms['train'])
print(len(images))
train_data,valid_data = torch.utils.data.dataset.random_split(images,[120000,29096])


train_dl = torch.utils.data.DataLoader(dataset=train_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])
valid_dl = torch.utils.data.DataLoader(dataset = valid_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])


#efficientnet = models.efficientnet_b4()
# efficientnet.classifier[1] = nn.Linear(in_features = 1792, out_features = 10, bias = True)
# model = efficientnet
# model.load_state_dict(torch.load(config['saved_path']))

from transformers import ViTImageProcessor, ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

model.classifier = torch.nn.Linear(in_features=768, out_features=10)

model = model.to(config['device'])

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train_model(model,criterion,optimizer,num_epochs=10):

    history = {}
    history['accuracy'],history['val_accuracy'] = [],[]
    history['loss'], history['val_loss'] = [], []
    batch_ct = 0
    example_ct = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        loss = []
        #Training
        model.train()
        run_corrects = 0
        for x,y in train_dl:
            x = x.to(config['device'])
            y = y.to(config['device'])

            optimizer.zero_grad()
            #optimizer.zero_grad(set_to_none=True)
            ######################################################################

            train_outputs = model(x)
            train_logits = train_outputs.logits
            _, train_preds = torch.max(train_logits, dim=1)
            train_loss = criterion(train_logits, y)
            run_corrects += torch.sum(train_preds == y.data)

            train_loss.backward() # Backpropagation this is where your W_gradient
            loss.append(train_loss.cpu().detach().numpy())

            optimizer.step() # W_new = W_old - LR * W_gradient
            example_ct += len(x)
            batch_ct += 1

        history['loss'].append(np.mean(np.array(loss)))

        #validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        # Disable gradient calculation for validation or inference using torch.no_rad()
        with torch.no_grad():
            for x,y in valid_dl:
                x = x.to(config['device'])
                y = y.to(config['device']) #CHW --> #HWC
                valid_outputs = model(x)
                valid_logits = valid_outputs.logits
                _, valid_preds = torch.max(valid_logits, dim=1)
                valid_loss = criterion(valid_logits,y)
                running_loss += valid_loss.item() * x.size(0)
                running_corrects += torch.sum(valid_preds == y.data)
                total += y.size(0)

        epoch_loss = running_loss / len(valid_data)
        epoch_acc = running_corrects.double() / len(valid_data)
        train_acc = run_corrects.double() / len(train_data)
        print("Train Accuracy",train_acc.cpu())
        print("Validation Loss is {}".format(epoch_loss))
        print("Validation Accuracy is {}".format(epoch_acc.cpu()))

        history['accuracy'].append(train_acc.cpu())
        history['val_accuracy'].append(epoch_acc.cpu())
        history['val_loss'].append(epoch_loss)

        if epoch_acc.cpu()>=max(history['val_accuracy']):
            torch.save(model.state_dict(), 'saved_models/vit_best.pt')
            print('Model weights updated!')

    torch.save(model.state_dict(), config['saved_path'])
    return history


history = train_model(model, criterion, optimizer, num_epochs=config['EPOCHS'])


plt.plot(range(config['EPOCHS']),history['accuracy'],label = 'Train Accuracy')
plt.plot(range(config['EPOCHS']),history['val_accuracy'],label = 'Validation Accuracy')
plt.legend()
plt.xlabel('EPOCHS')
plt.ylabel('Accuracy')
plt.title('ACCURACY CURVE')
plt.savefig('acc.png')
plt.show()
plt.clf()

plt.plot(range(config['EPOCHS']),history['loss'],label = 'Train Loss')
plt.plot(range(config['EPOCHS']),history['val_loss'],label = 'Validation Loss')
plt.legend()
plt.xlabel('EPOCHS')
plt.ylabel('Loss')
plt.title('LOSS CURVE')
plt.show()





