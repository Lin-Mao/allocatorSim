"""
Use torchrun script to run distributed training on single multi-gpu node with fault tolerance.
How to run script:
1. torchrun --standalone --nproc_per_node=gpu <script_name> <script_args>
2. torchrun --nproc_per_node=gpu --nnodes=1 --node_rank=0 --rdzv_id=457 --rdzv_endpoint=0.0.0.0:12355 <script_name> <script_args>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        print(f"[GPU{self.local_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)


class MyDataset(Dataset):
    # dedicated to MLPNet model
    def __init__(self, len):
        self.data = {}
        for i in range(len):
            self.data[i] = (torch.randn(20), torch.randn(1))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def load_train_objs():
    train_set = MyDataset(2048)
    model = MLPNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )

def ddp_setup():
    # torchrun will handle the environment variables
    # initialize the process group
    init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # set a device to this process


def main(total_epochs, batch_size):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single node train")
    parser.add_argument("--total_epochs", default=10, type=int, help="Total epochs to train the model")
    parser.add_argument("--batch_size", default=8, type=int, help="Input batch size on echo device (default: 32)")
    args = parser.parse_args()

    main(args.total_epochs, args.batch_size)
