import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


TRAIN_DATA_PATH = "../data/train_samples.win500.npy"
TEST_DATA_PATH = "../data/two_stage_test.npy"
REPORT_PATH = "../reports/nn_baseline.csv"
MODEL_PATH = "../models/nn_best.pth"
RANDOM_STATE = 57
BATCH_SIZE = 1024
EPOCHS = 30
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
INPUT_DIM = 1000
HIDDEN_DIM = 32
LEARNING_RATE = 1e-4


def read_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    data = pd.DataFrame(data.tolist())
    data = data[data.label.isin(['Sleep stage W', 'Sleep stage 4'])]
    return data


def create_features(data):
    y = data.label
    X = np.array(data.value.tolist())
    X = X.reshape((X.shape[0], 1000))
    person = data.person
    return X, y, person


def create_tt_split(X, y, person):
    train_p, test_p = train_test_split(person.unique(), test_size=0.2, random_state=RANDOM_STATE)
    print(train_p, test_p)
    return X[person.isin(train_p)], X[person.isin(test_p)], y[person.isin(train_p)], y[person.isin(test_p)]


def preprocess(data_path):
    data = read_data(data_path)
    X, y, person = create_features(data)
    ## normalize the data in order to resolve test bias
    X = ((X.T - X.mean(axis=1)) / X.std(axis=1)).T
    return X, y, person


def train_preprocess(data_path):
    X, y, person = preprocess(data_path)
    X_train, X_test, y_train, y_test = create_tt_split(X, y, person)
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test).squeeze()
    assert preds.shape == y_test.shape
    results = {
        'precision' : precision_score(y_test, preds, labels=y_test.unique(), average=None),
        'recall': recall_score(y_test, preds, labels=y_test.unique(), average=None)
    }
    return pd.DataFrame(results, index=y_test.unique())


def create_report(val_res, test_res, report_path):
    val_res['iter'] = 'val'
    test_res['iter'] = 'test'
    res = pd.concat([val_res, test_res], axis=0).set_index('iter', append=True).sort_index()
    res.to_csv(report_path)


class EEGDataset(Dataset):
    def __init__(self, X, y, scaler=None):
        super(EEGDataset, self).__init__()
        if scaler is None:
            scaler = StandardScaler().fit(X)
        self.scaler = scaler
        self.X = torch.as_tensor(scaler.transform(X), dtype=torch.float)
        self.y = torch.as_tensor(np.vstack([
            (y == 'Sleep stage 4').values,
            (y == 'Sleep stage W').values
        ]).T.astype(float), dtype=torch.float)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return self.y.shape[0]

    def get_scaler(self):
        return self.scaler


class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.p1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        p = F.dropout(x, p=0.2)
        x = F.dropout(x, p=0.2)
        p = F.celu(self.p1(p))
        x = F.gelu(self.l1(x))
        x = torch.cat((x, p), 1)
        x = self.l2(x)
        return x


def train_torch_model(X_train, y_train, X_val, y_val):
    train_dataset = EEGDataset(X_train, y_train)
    scaler = train_dataset.get_scaler()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = EEGDataset(X_val, y_val, scaler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    device = DEVICE
    model = SimpleModel(INPUT_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_mult = [y_train[y_train == "Sleep stage W"].shape[0] / y_train[y_train == "Sleep stage 4"].shape[0], 1]
    print(loss_mult)
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(loss_mult).to(device))
    best_val_loss = 1e10
    for epoch in trange(EPOCHS):
        model.train()
        train_history = []
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_fn(prediction, target)
            loss.backward()
            train_history.append(loss.cpu().detach().numpy().mean())
            optimizer.step()
        val_history = []
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(device)
                target = target.to(device)
                prediction = model(data)
                loss = loss_fn(prediction, target).cpu().numpy().mean()
                val_history.append(loss)
        print(epoch, np.mean(train_history), np.mean(val_history))
        if np.mean(val_history) < best_val_loss:
            best_val_loss = np.mean(val_history)
            with open(MODEL_PATH, "wb") as f:
                torch.save(model.state_dict(), f)
    with open(MODEL_PATH, "rb") as f:
        best_state_dict = torch.load(f, map_location=device)
        model.load_state_dict(best_state_dict)
    return model, scaler


def torch_evaluate(model, X_test, y_test, scaler):
    dataset = EEGDataset(X_test, y_test, scaler)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    preds = []
    device = DEVICE
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            preds.append(F.softmax(model(data), dim=1).cpu().numpy().round().astype(int)[:, 1].tolist())
    preds = np.array(["Sleep stage W" if item == 1 else "Sleep stage 4" for sublist in preds for item in sublist])
    assert preds.shape == y_test.shape
    results = {
        'precision' : precision_score(y_test, preds, labels=y_test.unique(), average=None),
        'recall': recall_score(y_test, preds, labels=y_test.unique(), average=None)
    }
    return pd.DataFrame(results, index=y_test.unique())


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = train_preprocess(TRAIN_DATA_PATH)
    model, scaler = train_torch_model(X_train, y_train, X_val, y_val)
    X_test, y_test, _ = preprocess(TEST_DATA_PATH)
    val_res = torch_evaluate(model, X_val, y_val, scaler)
    test_res = torch_evaluate(model, X_test, y_test, scaler)
    create_report(val_res, test_res, REPORT_PATH)
