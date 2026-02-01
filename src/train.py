import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RiceTypeClassificationBeginners.src.model import MyModel
from RiceTypeClassificationBeginners.src.dataset import dataset

def train_model(csv_path, batch_size=8, epochs=10, lr=0.01):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_df = pd.read_csv(csv_path)
    data_df.dropna(inplace=True) # 'inplace = True' drops any missing value 
    data_df.drop(['id'], axis=1, inplace=True) #axis=1 means it is column

    original_df=data_df.copy()
    for column in data_df.columns:
        data_df[column]=data_df[column]/data_df[column].abs().max()

    X=np.array(data_df.iloc[:,:-1]) #iloc means specific columns
    Y=np.array(data_df.iloc[:,-1]) #we are taking only the last column

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


    training_data=dataset(X_train, y_train)
    validation_data=dataset(X_val, y_val)
    testing_data=dataset(X_test, y_test)

    train_dataloader=DataLoader(training_data, batch_size=32, shuffle=True)
    val_dataloader=DataLoader(validation_data, batch_size=32, shuffle=True)
    test_dataloader=DataLoader(testing_data, batch_size=32, shuffle=True)

    model = MyModel(X.shape[1]).to(device)
    criterion=nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    for epoch in range(epochs):
        total_acc_train=0
        total_loss_train=0
        total_acc_val=0
        total_loss_val=0
        for data in train_dataloader:
            x, y = data
            prediction=model(x).squeeze(1)
            batch_loss=criterion(prediction, y)
            total_loss_train+=batch_loss.item()
            acc=(prediction.round()==y).sum().item()
            total_acc_train+=acc
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            for data in val_dataloader:
                x, y= data
                prediction=model(x).squeeze(1)
                batch_loss=criterion(prediction, y)
                total_loss_val+=batch_loss.item()
                acc=(prediction.round()==y).sum().item()
                total_acc_val+=acc
        total_loss_train_plot.append(round(total_loss_train/1000, 4))
        total_loss_validation_plot.append(round(total_loss_val/1000,4))

        total_acc_train_plot.append(round(total_acc_train/training_data.__len__()*100,4))
        total_acc_validation_plot.append(round(total_acc_val/validation_data.__len__()*100,4))
        print(f'''Epoch: {epoch+1} Train Loss {round(total_loss_train/1000, 4)} Train Acc:{round(total_acc_train/training_data.__len__()*100,4)}
            Validation Loss: {round(total_loss_val/1000,4)}''')
        print("="*25)

    torch.save(model.state_dict(), "rice_model.pth")
    print("Model saved as rice_model.pth")

    return model


