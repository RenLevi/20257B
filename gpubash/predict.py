import torch
import numpy as np
from dataset import DataLoader, ase_dataset_reader
from nn.network import EnergyNet, ForceNet
from nn.networkcombined import  combinedNet
from nn.utils import scatter
from sklearn.metrics import mean_squared_error
import json

def move_batch_to_device(batch_data_cpu: dict, device: torch.device) -> dict:
    batch_data_gpu = {}
    for key, value in batch_data_cpu.items():
        if isinstance(value, torch.Tensor):
            batch_data_gpu[key] = value.to(device, non_blocking=True if device.type == 'cuda' else False)
        else:
            batch_data_gpu[key] = value
    return batch_data_gpu

ase_kwargs = {'filename': 'Small_val.extxyz', 'format': 'extxyz'}
ase_kwargs_val = {'filename': 'Small_val.extxyz', 'format': 'extxyz'} 
batch_size = 16
atomicdata_kwargs, device, dl_kwargs, net_config = json.load(open('run/config.json', 'r'))
dataset = ase_dataset_reader(ase_kwargs, atomicdata_kwargs, device='cpu')

device = 'cuda:0'
device_ = torch.device(device)
if 'cuda' in device:
    torch.cuda.set_device(device_)
net = combinedNet(**net_config)
net.cuda()
net.load_state_dict(torch.load('run/train_all_best.pth'))
net.eval()
dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, **dl_kwargs)

size = len(dataset)
DFT_e = []
NN_e = []
DFT_e_peratom = []
NN_e_peratom = []
DFT_f = []
NN_f = []

DFT_q = []
NN_q = []

DFT_fermi = []
NN_fermi = []
DFT_fermishift = []
NN_fermishift = []

for ibatch, batch_data in enumerate(dataloader):
    batch_data = move_batch_to_device(batch_data, device_)
    results = net(batch_data)
    natoms = torch.bincount(batch_data["image_index"])
    natoms = natoms.view(-1, 1).float() 
    DFT_e.extend(list(torch.reshape(batch_data['image_energy'], (-1,)).cpu().numpy()))
    NN_e.extend(list(torch.reshape(results['image_energy'], (-1,)).detach().cpu().numpy()))
    DFT_e_peratom.extend(list(torch.reshape(batch_data['image_energy'].view(-1, 1) / natoms, (-1,)).cpu().numpy()))
    NN_e_peratom.extend(list(torch.reshape(results['image_energy'].view(-1, 1) / natoms, (-1,)).detach().cpu().numpy()))
    DFT_f.extend(list(torch.reshape(batch_data['atomic_forces'], (-1,)).cpu().numpy()))
    NN_f.extend(list(torch.reshape(results['forces'], (-1,)).cpu().numpy()))
    DFT_q.extend(list(torch.reshape(batch_data['charges'], (-1,)).cpu().numpy()))
    NN_q.extend(list(torch.reshape(results['charges'], (-1,)).detach().cpu().numpy()))
    DFT_fermi.extend(list(torch.reshape(batch_data['fermi'], (-1,)).cpu().numpy()))
    NN_fermi.extend(list(torch.reshape(results['fermi'], (-1,)).detach().cpu().numpy()))
    DFT_fermishift.extend(list(torch.reshape(batch_data['fermishift'], (-1,)).cpu().numpy()))
    NN_fermishift.extend(list(torch.reshape(results['fermishift'], (-1,)).detach().cpu().numpy()))
    
dataset_val = ase_dataset_reader(ase_kwargs_val, atomicdata_kwargs, device='cpu')
dataloader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size, **dl_kwargs)

DFT_ev = []
NN_ev = []
DFT_ev_peratom = []
NN_ev_peratom = []

DFT_fv = []
NN_fv = []

DFT_qv = []
NN_qv = []
DFT_fermiv = []
NN_fermiv = []
DFT_fermishiftv = []
NN_fermishiftv = []
for ibatch, batch_data in enumerate(dataloader):
    batch_data = move_batch_to_device(batch_data, device_)
    results = net(batch_data)
    natoms = torch.bincount(batch_data["image_index"])
    natoms = natoms.view(-1, 1).float() 
    DFT_ev.extend(list(torch.reshape(batch_data['image_energy'], (-1,)).cpu().numpy()))
    NN_ev.extend(list(torch.reshape(results['image_energy'], (-1,)).detach().cpu().numpy()))
    DFT_ev_peratom.extend(list(torch.reshape(batch_data['image_energy'].view(-1, 1) / natoms, (-1,)).cpu().numpy()))
    NN_ev_peratom.extend(list(torch.reshape(results['image_energy'].view(-1, 1) / natoms, (-1,)).detach().cpu().numpy()))
    DFT_fv.extend(list(torch.reshape(batch_data['atomic_forces'], (-1,)).cpu().numpy()))
    NN_fv.extend(list(torch.reshape(results['forces'], (-1,)).cpu().numpy()))
    DFT_qv.extend(list(torch.reshape(batch_data['charges'], (-1,)).cpu().numpy()))
    NN_qv.extend(list(torch.reshape(results['charges'], (-1,)).detach().cpu().numpy()))
    DFT_fermiv.extend(list(torch.reshape(batch_data['fermi'], (-1,)).cpu().numpy()))
    NN_fermiv.extend(list(torch.reshape(results['fermi'], (-1,)).detach().cpu().numpy()))
    DFT_fermishiftv.extend(list(torch.reshape(batch_data['fermishift'], (-1,)).cpu().numpy()))
    NN_fermishiftv.extend(list(torch.reshape(results['fermishift'], (-1,)).detach().cpu().numpy()))

import matplotlib.pyplot as plt

# energy
plt.scatter(DFT_e, NN_e, c='blue')
plt.scatter(DFT_ev, NN_ev, c='red')

plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_e), min(DFT_e), min(DFT_ev), min(NN_ev))
max_ = max(max(NN_e), max(DFT_e),max(NN_ev), max(DFT_ev))
plt.plot([min_, max_], [min_, max_])
plt.title('Energy (eV)')
RMSE = np.sqrt(mean_squared_error(DFT_e, NN_e))
RMSEv = np.sqrt(mean_squared_error(DFT_ev, NN_ev))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
print(f'Energy RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
plt.savefig("run/energy.png") 

# energy per atom
plt.figure()
plt.scatter(DFT_e_peratom, NN_e_peratom, c='blue')
plt.scatter(DFT_ev_peratom, NN_ev_peratom, c='red')

plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_e_peratom), min(DFT_e_peratom), min(DFT_ev_peratom), min(NN_ev_peratom))
max_ = max(max(NN_e_peratom), max(DFT_e_peratom),max(NN_ev_peratom), max(DFT_ev_peratom))
plt.plot([min_, max_], [min_, max_])
plt.title('Energy per atom (eV/atom)')
RMSE = np.sqrt(mean_squared_error(DFT_e_peratom, NN_e_peratom))
RMSEv = np.sqrt(mean_squared_error(DFT_ev_peratom, NN_ev_peratom))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
print(f'Energy per atom RMSE train: {RMSE:.3f} val: {RMSEv:.3f}  eV/atom')
plt.savefig("run/energy_peratom.png") 


# forces
plt.figure()
plt.scatter(DFT_f, NN_f, c='blue')
plt.scatter(DFT_fv, NN_fv, c='red')
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_f), min(DFT_f), min(DFT_fv), min(NN_fv))
max_ = max(max(NN_f), max(DFT_f),max(NN_fv), max(DFT_fv))
plt.plot([min_, max_], [min_, max_])
plt.title('Forces (eV/A)')
RMSE = np.sqrt(mean_squared_error(DFT_f, NN_f))
RMSEv = np.sqrt(mean_squared_error(DFT_fv, NN_fv))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
print(f'Force RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
plt.savefig("run/force.png") 
# charges 
plt.figure()
plt.scatter(DFT_q, NN_q, c='blue')
plt.scatter(DFT_qv, NN_qv, c='red')
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_q), min(DFT_q), min(DFT_qv), min(NN_qv))
max_ = max(max(NN_q), max(DFT_q),max(NN_qv), max(DFT_qv))
plt.plot([min_, max_], [min_, max_])
plt.title('charge (e)')
RMSE = np.sqrt(mean_squared_error(DFT_q, NN_q))
RMSEv = np.sqrt(mean_squared_error(DFT_qv, NN_qv))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
print(f'Charge RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
plt.savefig("run/charge.png") 

# fermi
plt.figure()
plt.scatter(DFT_fermi, NN_fermi, c='blue')
plt.scatter(DFT_fermiv, NN_fermiv, c='red')
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_fermi), min(DFT_fermi), min(DFT_fermiv), min(NN_fermiv))
max_ = max(max(NN_fermi), max(DFT_fermi),max(NN_fermiv), max(DFT_fermiv))
plt.plot([min_, max_], [min_, max_])
plt.title('fermi (V)')
RMSE = np.sqrt(mean_squared_error(DFT_fermi, NN_fermi))
RMSEv = np.sqrt(mean_squared_error(DFT_fermiv, NN_fermiv))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
print(f'Fermi levele RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
plt.savefig("run/fermi.png") 
# fermishift
plt.figure()
plt.scatter(DFT_fermishift, NN_fermishift, c='blue')
plt.scatter(DFT_fermishiftv, NN_fermishiftv, c='red')
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_fermishift), min(DFT_fermishift), min(DFT_fermishiftv), min(NN_fermishiftv))
max_ = max(max(NN_fermishift), max(DFT_fermishift),max(NN_fermishiftv), max(DFT_fermishiftv))
plt.plot([min_, max_], [min_, max_])
plt.title('fermishift (V)')
RMSE = np.sqrt(mean_squared_error(DFT_fermishift, NN_fermishift))
RMSEv = np.sqrt(mean_squared_error(DFT_fermishiftv, NN_fermishiftv))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
print(f'Fermi shift RMSE train: {RMSE:.3f} val: {RMSEv:.3f}')
plt.savefig("run/fermishift.png") 


