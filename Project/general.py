from Utils.databasify import PostgresDB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import joblib
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def clean_and_prepare(df, target_col, exclude_cols=[], one_hot=False, return_labels=False):
    df = df[df[target_col] != 'NaN'].copy()
    df.dropna(axis=1, how='all', inplace=True)

    # Drop columns with only 1 unique value (we don't need any columns that offer zero variance)
    df = df.loc[:, df.nunique() > 1]

    # Drop explicitly excluded columns
    df.drop(columns=exclude_cols, errors='ignore', inplace=True)

    if one_hot:
        categorical_cols = df.select_dtypes(include=['object']).columns.drop(target_col, errors='ignore')
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        label_encoders = None
    else:
        label_encoders = {}
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype == "object" and col != target_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

    # Fill numeric NaNs
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    if return_labels:
        return df, label_encoders
    return df

def train_decision_tree_model(df, target_col="disorder_subclass", exclude_cols=["genetic_disorder"],
                               max_depth=15, min_samples_split=2, min_samples_leaf=1, max_features=None):
    df, label_encoders = clean_and_prepare(df, target_col, exclude_cols, one_hot=False, return_labels=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print("\U0001F3AF Train Accuracy:", accuracy_score(y_train, train_preds))
    print("\U0001F3AF Test Accuracy:", accuracy_score(y_test, test_preds))
    print("\U0001F4CB Classification Report (Test):\n", classification_report(y_test, test_preds))

    joblib.dump(clf, "Models/mlproj/decision_tree_disorder.pkl")
    print("\U0001F4BE Model saved to 'decision_tree_disorder.pkl'")
    print('📊 Size of dataset:', df.shape)

    return clf, label_encoders

def train_mlp_model(df, target_col="disorder_subclass", exclude_cols=["genetic_disorder", "patient_id", "num"],
                    hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                    batch_size='auto', learning_rate='constant', max_iter=200, random_state=42):

    df = clean_and_prepare(df, target_col, exclude_cols, one_hot=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state
    )

    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print("\U0001F3AF Train Accuracy:", accuracy_score(y_train, train_preds))
    print("\U0001F3AF Test Accuracy:", accuracy_score(y_test, test_preds))
    print("\U0001F4CB Classification Report (Test):\n", classification_report(y_test, test_preds))

    joblib.dump(clf, "Models/mlproj/mlp_disorder_model.pkl")
    print("\U0001F4BE Model saved to 'mlp_disorder_model.pkl'")

    return clf

class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout=0.5, num_classes=9):
        super(TorchMLP, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_torch_mlp(df, target_col="disorder_subclass", exclude_cols=["genetic_disorder", "patient_id", "num"],
                    hidden_layers=[64, 32], dropout=0.5, lr=0.001, batch_size=64, epochs=30):

    df = clean_and_prepare(df, target_col, exclude_cols, one_hot=True)

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    classes, y = np.unique(y, return_inverse=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = TorchMLP(X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).argmax(dim=1)
        test_preds = model(X_test).argmax(dim=1)

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    print(f"\n✅ Torch MLP Train Accuracy: {train_acc:.4f}")
    print(f"✅ Torch MLP Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "Models/mlproj/torch_mlp_model.pt")
    print("💾 Torch model saved to 'Models/mlproj/torch_mlp_model.pt'")
    return model

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    db = PostgresDB()
    df = db.fetch_dataframe("SELECT * FROM raw_ml_data", table_name="raw_ml_data")
    train_torch_mlp(
        df,
        hidden_layers=[256, 256, 256],
        dropout=0.3,
        lr=0.0005,
        epochs=30  # with early stopping
    )