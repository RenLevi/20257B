import torch
import numpy as np
from dataset import DataLoader, ase_dataset_reader
from nn.networkcombined import  combinedNet
from nn.utils import scatter
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
import json
import os
import logging
import warnings
from torch.amp import autocast, GradScaler
import ase.data

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.*"
)

logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s',datefmt='[%Y/%m/%d %H:%M:%S]',level=logging.INFO, filename='run/train.log',
    filemode='w', )

loss_weight = {'energy': 1, 'force': 10, 'stress': 0.01, 'fermi': 0, 'fermishift': 0, 'charge': 0.1}


def move_batch_to_device(batch_data_cpu: dict, device: torch.device) -> dict:
    batch_data_gpu = {}
    for key, value in batch_data_cpu.items():
        if isinstance(value, torch.Tensor):
            batch_data_gpu[key] = value.to(device, non_blocking=True if device.type == 'cuda' else False)
        else:
            batch_data_gpu[key] = value
    return batch_data_gpu


def train_loop(dataloader, net, loss_fn, optimizer, scaler, batch_size, device):
    total_loss = 0
    size = len(dataset)
    loss_dict = {'energy': 0, 'force': 0, 'stress': 0, 'fermi': 0, 'fermishift': 0, 'charge': 0}
    total_Natom = 0
    net.train()
    for ibatch, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        batch_data = move_batch_to_device(batch_data, device)
        with autocast(device_type=device.type, dtype=torch.float16):
            results = net(batch_data)
            loss = {}
            Natom = len(batch_data['atomic_forces'])
            total_Natom += Natom
            loss['energy'] = loss_fn(results['image_energy'], batch_data['image_energy'].unsqueeze(-1))
            loss['force'] = loss_fn(results['forces'], batch_data['atomic_forces']) / 3 
            loss['stress'] = loss_fn(results['stress'].view(batch_size, -1), batch_data['stress'].view(batch_size, -1))
            loss['fermi'] = loss_fn(results['fermi'].squeeze(), batch_data['fermi'])
            loss['fermishift'] = loss_fn(results['fermishift'].squeeze(), batch_data['fermishift'])
            loss['charge'] = loss_fn(results['charges'], batch_data['charges'].unsqueeze(-1))
            batch_weighted_loss = sum(loss[key] * loss_weight[key] for key in loss_weight if key in loss)
        scaler.scale(batch_weighted_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ibatch % 50 == 0:
            current = (ibatch + 1) * batch_size
            logging.info(f" {current / size * 100:5.2f}%    loss: {batch_weighted_loss/batch_size:>7f}")
        for key in loss_dict.keys():
            loss_dict[key] += loss[key].item()
        total_loss += batch_weighted_loss.item()
    loss_dict['force'] = loss_dict['force'] / total_Natom
    loss_dict['charge'] = loss_dict['charge'] / total_Natom
    loss_dict['energy'] = loss_dict['energy'] / size
    loss_dict['fermi'] = loss_dict['fermi'] / size
    loss_dict['fermishift'] = loss_dict['fermishift'] / size


    return total_loss / size, loss_dict  # per image loss



def valid_loop(dataloader, net, loss_fn, optimizer, batch_size, device):
    net.eval()  
    total_loss = 0
    size = len(dataset_val)
    loss_dict = {'energy': 0, 'force': 0, 'stress': 0, 'fermi': 0, 'fermishift': 0, 'charge': 0}
    total_Natom = 0
    for ibatch, batch_data in enumerate(dataloader):  
        batch_data = move_batch_to_device(batch_data, device)
        with autocast(device_type=device.type, dtype=torch.float16):
            results = net(batch_data)
            loss = {}
            Natom = len(batch_data['atomic_forces'])
            total_Natom += Natom
            loss['energy'] = loss_fn(results['image_energy'], batch_data['image_energy'].unsqueeze(-1))
            loss['force'] = loss_fn(results['forces'], batch_data['atomic_forces']) / 3 
            loss['stress'] = loss_fn(results['stress'].view(batch_size, -1), batch_data['stress'].view(batch_size, -1))
            loss['fermi'] = loss_fn(results['fermi'].squeeze(), batch_data['fermi'])
            loss['fermishift'] = loss_fn(results['fermishift'].squeeze(), batch_data['fermishift'])
            loss['charge'] = loss_fn(results['charges'], batch_data['charges'].unsqueeze(-1))
        batch_weighted_loss = sum(loss[key] * loss_weight[key] for key in loss_weight if key in loss)
        for key in loss_dict.keys():
            loss_dict[key] += loss[key].item()
        total_loss += batch_weighted_loss.item()
    average_loss = total_loss / size
    loss_dict['force'] = loss_dict['force'] / total_Natom
    loss_dict['charge'] = loss_dict['charge'] / total_Natom
    loss_dict['energy'] = loss_dict['energy'] / size
    loss_dict['fermi'] = loss_dict['fermi'] / size
    loss_dict['fermishift'] = loss_dict['fermishift'] / size

    return average_loss, loss_dict


if __name__ == '__main__':
    ase_kwargs = {'filename': 'middle_train.extxyz', 'format': 'extxyz'}   # WF dataset
    ase_kwargs_val = {'filename': 'middle_val.extxyz', 'format': 'extxyz'} 
    if not os.path.exists('run'):
        os.makedirs('run')
    checkpoint_name = 'run/train_all.pth'       # pretrained model
    atomicdata_kwargs = {'r_max': 5.0}
    device = 'cuda:0'
    epoch_size = 1000
    batchsize = 32
    device_ = torch.device(device)
    if 'cuda' in device:
        torch.cuda.set_device(device_)
    dl_kwargs = dict(atomic_map=['Cu', 'C', 'O', 'H'], device=device, WF=True, charge=True,)
    dataset = ase_dataset_reader(ase_kwargs, atomicdata_kwargs, device='cpu')
    dataloader_train = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=16, pin_memory=True, drop_last=True, **dl_kwargs)
    logging.info(f"train dataset initialed in cpu")
    dataset_val = ase_dataset_reader(ase_kwargs_val, atomicdata_kwargs, device='cpu')
    dataloader_val =  DataLoader(dataset=dataset_val, batch_size=batchsize, shuffle=True, num_workers=16, pin_memory=True, drop_last=True, **dl_kwargs)
    logging.info(f"val dataset initialed in cpu")
    config = 'run/config.json'

    
    net_config = dict(
        irreps_embedding_out='16x0e',
        irreps_conv_out='32x0e',
        r_max=atomicdata_kwargs['r_max'],
        num_layers1=4,
        num_layers2=0,
        num_basis=8,
        cutoff_p=6,
        hidden_mul=32,
        lmax=2,
        convolution_kwargs={'invariant_layers': 2, 'invariant_neurons': 64, 'avg_num_neighbors': 20, 'use_sc': True},
        databased_setting={'element_number': len(dl_kwargs['atomic_map']),'atomic_numbers':[ase.data.atomic_numbers[symbol] for symbol in dl_kwargs['atomic_map']],  'atomic_scale': 1.0, 'atomic_shift': -3.57},    # scale = sigma   ; shift = u  ; u is the average value while sigma is the standard deviation
        charge=True,
        device=device,
        dFdQ=True
    )

    net = combinedNet(**net_config)
    config = (atomicdata_kwargs, device, dl_kwargs, net_config)
    json.dump(config, open('run/config.json', 'w'))
    if 'cuda' in device:
        net.cuda()
    net.load_state_dict(torch.load('run/train_all_best.pth'))
    loss_fn = torch.nn.HuberLoss(delta=1.0, reduction='sum')     # Huber loss
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      
    factor=0.5,      
    patience=10,        
    min_lr=1e-7,        
    verbose=True        
    )
    scaler = GradScaler(enabled=device_.type == 'cuda')

    loss_his = []
    valid_loss_his = []
    valid_min = 9999
    device = torch.device(device)
    for epoch in range(epoch_size):
        logging.info(f"Epoch {epoch + 1}: ")
        loss_, loss_dict_train = train_loop(dataloader_train, net, loss_fn, optimizer, scaler, batch_size=batchsize, device=device)
        loss_val, loss_dict_val = valid_loop(dataloader_val, net, loss_fn, optimizer, batch_size=batchsize, device=device)
        scheduler.step(loss_val)
        logging.info(f"Epoch  {epoch + 1}  train_loss  {loss_:.2E};   valid_loss  {loss_val:.2E}     E_train {loss_dict_train['energy']:.2E}   E_val {loss_dict_val['energy']:.2E} F_tain {loss_dict_train['force']:.2E}  F_val {loss_dict_val['force']:.2E}   Q_train {loss_dict_train['charge']:.2E}  Q_val {loss_dict_val['charge']:.2E} fermi_train {loss_dict_train['fermi']:.2E}  fermi_val {loss_dict_val['fermi']:.2E} ")
    
        if epoch % 1 == 0:
            torch.save(net.state_dict(), checkpoint_name)  # backup
        if loss_val < valid_min:
            checkpoint_bk = checkpoint_name[:-4] + '_best.pth'
            torch.save(net.state_dict(), checkpoint_bk)
            valid_min = loss_val
        loss_his += [loss_]
        valid_loss_his += [loss_val]

    torch.save(net.state_dict(), checkpoint_name)
    logging.info(loss_his)
