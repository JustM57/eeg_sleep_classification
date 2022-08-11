import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


TRAIN_DATA_PATH = "../data/train_samples.win500.npy"
TEST_DATA_PATH = "../data/two_stage_test.npy"
REPORT_PATH = "../reports/catboost_baseline.csv"
RANDOM_STATE = 57


def read_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    data = pd.DataFrame(data.tolist())
    data = data[data.label.isin(['Sleep stage W', 'Sleep stage 4'])]
    print(data.iloc[0])
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


def train_model(X_train, y_train, X_test, y_test):
    model = CatBoostClassifier(
        iterations=1000,
        random_seed=RANDOM_STATE,
        loss_function='MultiClass',
        auto_class_weights='Balanced',
        learning_rate=0.05
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=10,
        plot=False
    )
    return model


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


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = train_preprocess(TRAIN_DATA_PATH)
    model = train_model(X_train, y_train, X_val, y_val)
    X_test, y_test, _ = preprocess(TEST_DATA_PATH)
    val_res = evaluate(model, X_val, y_val)
    test_res = evaluate(model, X_test, y_test)
    create_report(val_res, test_res, REPORT_PATH)
