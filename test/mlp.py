import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

class Net(nn.Module):
    def __init__(self, hidden_size: int = 10):
        super(Net, self).__init__()
        self.linear_statck = nn.Sequential(
            nn.Linear(20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear_statck(x)


class MyDataset(Dataset):
    def __init__(self, size: int = 1000):
        super(MyDataset, self).__init__()
        import random
        random.seed(10)
        self.data = {}
        for i in range(size):
            self.data[i] = (torch.randn(20), torch.randn(5))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Step {batch:>4d} loss: {loss.item():>5f}")


def main():
    model = Net(10000).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3)

    dataset = MyDataset(10000)
    train_dataloader = DataLoader(dataset, batch_size=16)

    train(train_dataloader, model, loss_fn, optimizer)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()