import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import warnings

class TripletDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.class_to_indices = self._build_class_index()

    def _build_class_index(self):
        class_map = {}
        for idx, label in enumerate(self.labels):
            label_int = label.item() if isinstance(label, torch.Tensor) else label
            class_map.setdefault(label_int, []).append(idx)
        return class_map

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        anchor = self.embeddings[idx]
        anchor_label = self.labels[idx].item() if isinstance(self.labels[idx], torch.Tensor) else self.labels[idx]

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.class_to_indices[anchor_label])
        positive = self.embeddings[pos_idx]

        neg_label = random.choice([l for l in self.class_to_indices.keys() if l != anchor_label])
        neg_idx = random.choice(self.class_to_indices[neg_label])
        negative = self.embeddings[neg_idx]

        return anchor, positive, negative

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64], embedding_dim=32, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def clean_and_prepare(df, target_col, exclude_cols=[]):
    df = df[df[target_col] != 'NaN'].copy()
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.nunique() > 1]
    df.drop(columns=exclude_cols, errors='ignore', inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns.drop(target_col, errors='ignore')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def train_triplet_and_classify_with_gmm(df, target_col="disorder_subclass", exclude_cols=["genetic_disorder", "patient_id", "num"],
                                        hidden_layers=[128, 64], embedding_dim=32, dropout=0.3, lr=1e-3,
                                        batch_size=64, epochs=20, n_components=7):
    df = clean_and_prepare(df, target_col, exclude_cols)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    classes, y = np.unique(y, return_inverse=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TripletDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingMLP(X_train.shape[1], hidden_layers, embedding_dim, dropout)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for anchor, positive, negative in train_loader:
            optimizer.zero_grad()
            out_a = model(anchor)
            out_p = model(positive)
            out_n = model(negative)
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Triplet Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_embeddings = model(X_test_tensor).numpy()

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm_labels = gmm.fit_predict(test_embeddings)

    ari = adjusted_rand_score(y_test, gmm_labels)
    print(f"\n📊 Adjusted Rand Index (GMM vs True): {ari:.4f}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(test_embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gmm_labels, cmap='tab10', s=10)
    plt.title("t-SNE of Triplet Embeddings + GMM Clusters")
    plt.savefig("models/mlproj/tsne_gmm_triplet.png")
    print("📈 t-SNE plot saved to models/mlproj/tsne_gmm_triplet.png")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    from Utils.databasify import PostgresDB
    db = PostgresDB()
    df = db.fetch_dataframe("SELECT * FROM raw_ml_data", table_name="raw_ml_data")
    train_triplet_and_classify_with_gmm(df, hidden_layers=[256, 512, 512, 512], embedding_dim=1024, dropout=0.2, epochs=50)
