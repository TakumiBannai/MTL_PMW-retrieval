import datetime
import pickle
import torch
import torch.nn as nn
import torch.optim as opt
import joblib
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from data import load_data, MeasureDataset
from model import *
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loss import FocalLoss, ExpandMSELoss, CombinedLoss, WES
from utils import *

def retrieve_pred_label_single(type, model, dataloader, rain_th, log_transy=False):
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            if log_transy == True:
                y[:, 1] = torch.log(y[:, 1] + 1)
            # Get pred
            output = model(x.to(device))
            # Get label
            if type=="Clsf":
                # y = y[:,0].long()  OLD
                y = convert_ClsfLabel(rain_th = rain_th, y = y)
            if type=="Reg":
                y = y[:,1]
            # Collect data
            pred.append(output)
            label.append(y)
    # To tensor
    pred = torch.cat(pred, axis = 0)
    label = torch.cat(label, axis = 0)
    # To Binary
    if type=="Clsf":
        pred = torch.argmax(pred, axis=1)
    # Inv-trans
    if type=="Reg":
        if log_transy == True:
            # label-inv
            label = torch.exp(label) - 1
            # pred-inv
            pred = torch.exp(pred) - 1
    return pred.cpu().detach().numpy(), label.cpu().detach().numpy()


def retrieve_pred_label(model, dataloader, rain_th, log_transy=False):
    model.eval()
    pred_clsf = []
    pred_reg = []
    label_clsf = []
    label_reg = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            if log_transy == True:
                y[:, 1] = torch.log(y[:, 1] + 1)
            # Get pred
            pred_clsf_, pred_reg_ = model(x.to(device))
            # Get label
            label_reg_ = y[:, 1]
            # Convert Clsf-label
            label_clsf_ = convert_MutiLabel(rain_th = rain_th, y = y)
            # Collect data
            pred_clsf.append(pred_clsf_)
            pred_reg.append(pred_reg_)
            label_clsf.append(label_clsf_)
            label_reg.append(label_reg_)
    # To tensor
    pred_clsf = torch.cat(pred_clsf, axis = 0)
    pred_reg = torch.cat(pred_reg, axis = 0)
    label_clsf = torch.cat(label_clsf, axis = 0)
    label_reg = torch.cat(label_reg, axis = 0)
    # To Binary
    pred_clsf = torch.argmax(pred_clsf, axis=1)
    # Inv-trans
    if log_transy == True:
        # label-inv
        label_reg = torch.exp(label_reg) - 1
        # pred-inv
        pred_reg = torch.exp(pred_reg) - 1
    return pred_clsf.cpu().detach().numpy(), pred_reg.cpu().detach().numpy(), label_clsf.cpu().detach().numpy(), label_reg.cpu().detach().numpy()


def train_single_model(type, model, lr, n_EPOCH, clsf_loss_func, reg_loss_func, show_progress=True, rain_th=0.0, log_transy=False):
    if type=="Clsf":
        if clsf_loss_func == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif clsf_loss_func == "FocalLoss":
            criterion = FocalLoss(classes=2,alpha=torch.FloatTensor([1,1]).to(device),size_average=False).to(device)
    if type=="Reg":
        if reg_loss_func == "MSELoss":
            criterion = nn.MSELoss()
        elif reg_loss_func == "WES":
            criterion = WES(beta=3, label_numpy=precipitation_sample)
    optimizer = opt.Adam(model.parameters(), lr)
    # Train model
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(n_EPOCH)):
        # Train loop
        model.train()
        train_batch_loss = []
        for idx, (x, y) in enumerate(train_dataloader):
            # Log-trans(y)
            if log_transy == True:
                y[:, 1] = torch.log(y[:, 1] + 1)
            optimizer.zero_grad()
            output = model(x.to(device))
            if type=="Clsf":
                # y = y[:,0].long() # OLD
                y = convert_ClsfLabel(rain_th = rain_th, y = y).long()
            if type=="Reg":
                y = y[:,1].unsqueeze(1)
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
        # Val loop
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(val_dataloader):
                # Log-trans(y)
                if log_transy == True:
                    y[:, 1] = torch.log(y[:, 1] + 1)
                if type=="Clsf":
                    # y = y[:,0].long()# OLD
                    y = convert_ClsfLabel(rain_th = rain_th, y = y).long()
                if type=="Reg":
                     y = y[:,1].unsqueeze(1)
                output = model(x.to(device))
                loss = criterion(output, y.to(device))
                val_batch_loss.append(loss.item())
        train_loss.append(np.mean(train_batch_loss))
        val_loss.append(np.mean(val_batch_loss))
        # progress option
        if show_progress == True:
            if epoch % 5 == 0:
                print("Epoch. {} -----------------------------------".format(epoch))
                print("Train loss: {a:.3f}, Val loss: {b:.3f}".format(a = train_loss[-1], b = val_loss[-1]))

    # tain-loss visualization option
    if show_progress == True:
        plt.figure(figsize=(6, 5))
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.title('Training loss')
        plt.legend()

    # result option
    if show_progress == True:
        # Retrieve pred and lable
        pred, label = retrieve_pred_label_single(type, model, test_dataloader, rain_th, log_transy)
        if type=="Clsf":
            # 評価対象外の項目は乱数設定
            evaluation = EvaluationIndices(pred_reg=np.random.rand(5), label_reg=np.random.rand(5),
                                           pred_cls=pred, label_cls=label)
        if type=="Reg":
            # 評価対象外の項目は乱数設定
            evaluation = EvaluationIndices(pred_reg=pred, label_reg=label,
                                           pred_cls=np.array([0, 1, 0, 1, 0]), label_cls=np.array([0, 1, 0 ,1, 1]))
        print(evaluation.evaluate())

    # 2d-scatter plot option
    if show_progress == True:
        if type=="Reg":
            pred, label = retrieve_pred_label_single(type, model, test_dataloader, rain_th, log_transy)
            show_regression_performance_2d(pred, label, title_name="Single-task)" + reg_loss_func)

    # Binned-result
    if show_progress == True:
        # Retrieve pred and lable
        if type=="Reg":
            # Retrieve pred and lable
            pred, label = retrieve_pred_label_single(type, model, test_dataloader, rain_th, log_transy)
            print(compute_index_comparison(pred, label, out_type='df'))

    return model


def train_multitask_model(model, lr, n_EPOCH, clsf_loss_func, reg_loss_func, show_progress=True, rain_th=0.0, log_transy=False, pr=0.5, class_weights=[0.05, 0.1, 0.15, 0.5, 0.2]):
    strategy = {
    "clsf_begin_epoch":0,"clsf_end_epoch":20,
    "reg_begin_epoch":n_EPOCH,"reg_end_epoch":n_EPOCH+10,
    "mix_begin_epoch":20,"mix_end_epoch":n_EPOCH,
    }
    # Loss function for classification
    if clsf_loss_func == "CrossEntropyLoss":
        clsf_loss = nn.CrossEntropyLoss()
    elif clsf_loss_func == "WeigtedCrossEntropy":
        clsf_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights[:4], dtype=torch.float))
    elif clsf_loss_func == "FocalLoss":
        clsf_loss = FocalLoss(classes=2,alpha=torch.FloatTensor([1,1]).to(device),size_average=False).to(device)
    # Loss function for regression
    if reg_loss_func == "MSELoss":
        reg_loss = nn.MSELoss()
    elif reg_loss_func == "WES":
        reg_loss = WES(beta=3, label_numpy=precipitation_sample)
    # Loss & Optimizer
    criterion = CombinedLoss(reg_loss=reg_loss, clsf_loss=clsf_loss, strategy=strategy, alpha=10, class_weights=class_weights).to(device)
    optimizer = opt.Adam(model.parameters(), lr)
    # Train model
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(n_EPOCH)):
        # Train loop
        model.train()
        train_batch_loss = []
        for idx, (x, y) in enumerate(train_dataloader):
            # Log-trans(y)
            if log_transy == True:
                y[:, 1] = torch.log(y[:, 1] + 1)
            # Convert clsf-label
            y[:, 0] = convert_MutiLabel(rain_th = rain_th, y = y)
            optimizer.zero_grad()
            clsf, reg = model(x.to(device))
            loss, loss_type = criterion(clsf, reg, y.to(device), epoch, True)
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
        # Val loop
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(val_dataloader):
                # Log-trans(y)
                if log_transy == True:
                    y[:, 1] = torch.log(y[:, 1] + 1)
                # Convert clsf-label
                y[:, 0] = convert_MutiLabel(rain_th = rain_th, y = y)
                clsf, reg = model(x.to(device))
                loss, loss_type = criterion(clsf, reg, y.to(device), epoch, True)
                val_batch_loss.append(loss.item())
        train_loss.append(np.mean(train_batch_loss))
        val_loss.append(np.mean(val_batch_loss))
        # progress option
        if show_progress == True:
            if epoch % 5 == 0:
                print("Epoch. {} -----------------------------------".format(epoch))
                print("Train loss: {a:.3f}, Val loss: {b:.3f}".format(a = train_loss[-1], b = val_loss[-1]))

    # visualization optin
    if show_progress == True:
        plt.figure(figsize=(6, 5))
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.title('Training loss')
        plt.legend()
    
    # 2d-scatter plot option
    if show_progress == True:
        # 2d-scatter plot
        test_pred_clsf, test_pred_reg, test_label_clsf, test_label_reg = retrieve_pred_label(model, test_dataloader, rain_th, log_transy)
        show_regression_performance_2d(test_pred_reg, test_label_reg, title_name="Two-task)" + clsf_loss_func + " & " + reg_loss_func)
    
    # Binned-result
    if show_progress == True:
        # Retrieve pred and lable
        test_pred_clsf, test_pred_reg, test_label_clsf, test_label_reg = retrieve_pred_label(model, test_dataloader, rain_th, log_transy)
        print(compute_index_comparison(test_pred_reg, test_label_reg, out_type='df'))

    return model

