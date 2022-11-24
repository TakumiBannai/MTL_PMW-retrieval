import joblib
from data import load_data, MeasureDataset
from model import *
from train import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("dataset/train.pt","rb") as f:
    train_dataset = joblib.load(f)

with open("dataset/valid.pt","rb") as f:
    valid_dataset = joblib.load(f)

with open("dataset/test.pt","rb") as f:
    test_dataset = joblib.load(f)

train_dataloader = DataLoader(train_dataset,
                                batch_size=1024,
                                shuffle=True,
                                drop_last=False,
                                num_workers=2,
                                worker_init_fn=worker_init_fn)

val_dataloader = DataLoader(valid_dataset,
                                batch_size=1024,
                                shuffle=True,
                                drop_last=False,
                                num_workers=2,
                                worker_init_fn=worker_init_fn)

test_dataloader = DataLoader(test_dataset,
                                batch_size=1024,
                                shuffle=False,
                                drop_last=False,
                                worker_init_fn=worker_init_fn
                                )

# Train model
model = CNNSingleTask("Clsf").to(device)ß
trained_model = train_single_model(type="Clsf", model=model, lr=1e-5, n_EPOCH=100,
                            clsf_loss_func="CrossEntropyLoss", reg_loss_func="MSELoss",
                            show_progress=True, rain_th=0.0, log_transy=False)

model = CNNSingleTask("Reg").to(device)
trained_model = train_single_model(type="Reg", model=model, lr=1e-5, n_EPOCH=100,
                            clsf_loss_func="CrossEntropyLoss", reg_loss_func="MSELoss",
                            show_progress=True, rain_th=0.0, log_transy=False)


model = CNNMultiTask().to(device)
trained_model = train_multitask_model(model=model, lr=1e-5, n_EPOCH=100,
                            clsf_loss_func="WeigtedCrossEntropy", reg_loss_func="MSELoss",
                            show_progress=True, rain_th=[0.1, 1.0, 10.0], class_weights=[0.05, 0.1, 0.15, 0.5, 0.2])


