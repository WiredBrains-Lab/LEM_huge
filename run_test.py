import os, json, time
import torch
from torch.utils.data import DataLoader,DistributedSampler
import lem
import deepspeed
import argparse

project_base = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ds_config = os.path.join(project_base,"ds_config.json")

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank pasdf -hsed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()

torch.cuda.set_device(cmd_args.local_rank)

deepspeed.init_distributed()

world_size = deepspeed.comm.get_world_size()
rank = deepspeed.comm.get_rank()

print(f"world_size: {world_size}, rank: {rank}")

with open(ds_config, 'r') as f:
    config = json.load(f)
batch_size = config['train_batch_size']

train_n = 10000
test_n = batch_size

if rank==0:
    train_dataset = lem.test_data.FakeEEGDataset(num_samples=train_n,mmap_path=f'data/train_data.dat',create=True)
    test_dataset = lem.test_data.FakeEEGDataset(num_samples=test_n,mmap_path=f'data/test_data.dat',create=True)
else:
    time.sleep(1)
deepspeed.comm.barrier()

if rank!=0:
    train_dataset = lem.test_data.FakeEEGDataset(num_samples=train_n,mmap_path=f'data/train_data.dat',create=False)
    test_dataset = lem.test_data.FakeEEGDataset(num_samples=test_n,mmap_path=f'data/test_data.dat',create=False)

print(f"train_dataset: {train_dataset.data.shape}")

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    sampler    = train_sampler,
    num_workers= 0,
    pin_memory = False,
    drop_last  = True
)

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"train_loader: {len(train_loader)}, test_loader: {len(test_loader)}")

lem.transformer.train_transformer(
    train_loader=train_loader,
    val_loader=test_loader,
    bin_size=10,
    lr=5e-5,
    lr_ramp=50,
    lr_cycle=100,
    lr_min=1e-6,
    weight_decay=1e-5,
    num_epochs=10000,
    conv_dims=[16, 32, 64],
    emb_dim=1024,
    num_layers=12,
    num_experts=96,
    dim_feedforward=4096,
    top_k=1,
    decoder_hidden=[1024, 1024, 1024],
    device=cmd_args.local_rank,
    args=cmd_args
)