import torch
import random
import numpy as np
from dataset import *


from imgSimul import ImgSimulator
from torch.utils.data import  DataLoader 
from  psfSimul import psfSimulator
from ArgParser import Argument 
from torch.nn import functional as F

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import Rbf
from proxyModel import *

import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator
from matplotlib.patches import Circle

import lpips 
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

import matplotlib
matplotlib.use('Agg') 

FDTD_result = {}

class GaussianNoise:
    def __init__(self, mean, stddev, device):
        self.mean = mean
        self.stddev = stddev
        self.device = device

    def sample(self, size):
        noise = torch.zeros(size, device=self.device).normal_(self.mean, self.stddev)
        return noise

def sub_show(patches, dest):
    num_images = patches.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images*2, 2))
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(patches[i].detach().cpu().numpy())
        ax.axis('off')
    plt.tight_layout()
    #plt.show() 
    plt.savefig(dest)
    #plt.close()

def get_2d_diameter(proxy, phase_map):
    diameter = proxy.get_diameter(phase_map) * 280 + 90 
    diameter = torch.round(diameter)



class RBF:
    def __init__(self, mat_path, name, rbf_function='gaussian', epsilon=2.0):
        data = loadmat(mat_path)
        self.z = data[name]
        self.d_grid = np.linspace(0, 1, 26)
        self.a_grid = np.linspace(0, 1, 21)
        
        self.function = rbf_function 
        self.epsilon = epsilon 
        
        D, A = np.meshgrid(self.d_grid, self.a_grid, indexing='ij')  # same shape as z
        self.rbf = Rbf(D.flatten(), A.flatten(), self.z.flatten(),
                       function=self.function, epsilon=self.epsilon)
        
    def forward(self, d_query, a_query):
        return self.rbf(d_query, a_query)
    
class ScatteredInterpolator:
    def __init__(self, mat_path, name):
        data = loadmat(mat_path)
        z = data[name]

        if z.shape[0] == 21 and z.shape[1] == 26:
            z = z.T  # ensure (d, a) shape

        self.z = z
        self.d_grid = np.linspace(0, 1, z.shape[0])
        self.a_grid = np.linspace(0, 1, z.shape[1])

        D, A = np.meshgrid(self.d_grid, self.a_grid, indexing='ij')
        coords = np.stack([D.flatten(), A.flatten()], axis=1)
        values = z.flatten()

        self.interp = LinearNDInterpolator(coords, values)

    def forward(self, d_query, a_query):
        #d_query = np.asarray(d_query).flatten()
        #a_query = np.full_like(d_query, fill_value=a_query)
        query_pts = np.stack([d_query, a_query], axis=1)
        return self.interp(query_pts)
    
class FDTD_interp:
    def __init__(self, device):
        self.TM_proxy = ScatteredInterpolator('./data/fdtd/Tx(TM).mat', 'Tx')
        self.TE_proxy = ScatteredInterpolator('./data/fdtd/Ty(TE).mat', 'Ty')
        self.PM_proxy = ScatteredInterpolator('./data/fdtd/phaseEx(TM).mat', 'phaseEx')
        self.PE_proxy = ScatteredInterpolator('./data/fdtd/phaseEy(TE).mat', 'phaseEy')
        
        self.device = device
        
    def forward(self, d_query, a_query):
        
        d_array = np.asarray(d_query).flatten() 
        a_array = np.full_like(d_array, fill_value=a_query)
        
        
        
        T_tm = self.TM_proxy.forward(d_array, a_array).reshape(1, -1)
        T_te = self.TE_proxy.forward(d_array, a_array).reshape(1, -1)
        P_tm = self.PM_proxy.forward(d_array, a_array).reshape(1, -1)
        P_te = self.PE_proxy.forward(d_array, a_array).reshape(1, -1)
        
        A_tm, A_te = np.sqrt(np.clip(T_tm, 0, 1.)), np.sqrt(np.clip(T_te, 0, 1.))
        
        return [torch.from_numpy(A_tm).to(device), torch.from_numpy(P_tm).to(device), torch.from_numpy(A_te).to(device), torch.from_numpy(P_te).to(device)]

def get_psfs(args, phase_map, v_theta, proxy, fdtd_proxy):
    k = args.k 
    patches = []
    r_list = []
    fe_list = []
    fe_na_list = []
    psf_list = None
    

    for i in range(v_theta.shape[0]):
        angle = v_theta[i]
        y = -args.f * torch.sin(torch.deg2rad(v_theta[i]))
        x = torch.zeros_like(y)
        offset = [y, x]
        
        #result_FDTD = proxy.forward(phase_map, angle/50.)
        
        use_fdtd = True 
        if use_fdtd: 
            diameters = proxy.ph2duty.forward(phase_map)  
            result_FDTD = fdtd_proxy.forward(diameters.cpu().numpy(), angle.item()/50.)
        else :
            diameters = proxy.ph2duty.forward(phase_map)
            _, phase_zero,  _, _ = fdtd_proxy.forward(diameters.cpu().numpy(), 0)
            phase_tm = phase_zero.clone() 
            phase_te = phase_zero.clone() 
            Amp_tm = torch.ones_like(phase_tm)
            Amp_te = torch.ones_like(phase_te)
            result_FDTD = [Amp_tm, phase_tm, Amp_te, phase_te]
            
        
        psf, input_light = psf_simulator.forward2(result_FDTD, args.sensor_dist, offset, args.wvl, -1*angle, mode='far')

        if psf_list is None: 
            psf_list = psf 
        else: 
            psf_list += psf 
            
        test_array = [t.cpu().numpy() for t in result_FDTD]
        FDTD_result['%d'%(i*10)] = test_array
        
        
        peak_idx = torch.nonzero(psf == psf.max())[0]
        cx, cy = peak_idx[-1], peak_idx[-2]
        _, _, h, w = psf.shape
        kk = 30
        patch_eval = psf[0][0][cy-kk:cy+kk, cx-kk:cx+kk] 

        
        N = patch_eval.shape[0]
        peak_val = torch.max(patch_eval)
        peak_idx = torch.nonzero(patch_eval == peak_val)[0]
        cy, cx = peak_idx[0].item(), peak_idx[1].item() 
        
        y, x = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
        r = torch.sqrt((x-cx)**2 + (y-cy)**2)
        r_flat = r.flatten() 
        psf_flat = patch_eval.flatten() 
        
        sorted_idx = torch.argsort(r_flat)
        r_sorted = r_flat[sorted_idx]
        psf_sorted = psf_flat[sorted_idx]
        
        psf_sorted_norm = psf_sorted / peak_val
        half_max_idx = torch.where(psf_sorted_norm <= 0.5)[0][0]
        fwhm_radius = r_sorted[half_max_idx].item() 
        
        # fwhm
        radius_m = fwhm_radius * 400e-9 
        diameter_m = 2 * radius_m 
        
        NA = 0.78 
        wavelength = 850e-9 
        pixel_pitch = 400e-9 
        airy_radius_m = 0.61 * wavelength / NA * 3
        airy_radius_px = airy_radius_m / pixel_pitch 
        region_radius = fwhm_radius* 3 #airy_radius_px 
        
        mask = ((x-cx)**2 + (y-cy)**2) <= region_radius ** 2 
        region_intensity = patch_eval[mask].sum().item() 
        
        fe = region_intensity / input_light.sum()/0.64
        print("FE %d: "%(i * 10), fe.item() * 100., "%")
        fe_list.append(fe)
        
    
    fe_list = torch.stack(fe_list)
    
    return fe_list
 
 



if __name__ == "__main__":
    parser = Argument() 
    args = parser.parse()
    
    device = torch.device(args.device)
    ref_wvl = args.wvl

    noise_model = GaussianNoise(0, 2e-2, device)
    # dataset
    dataset_train = MyDataset(args.train_dataset_path, 'train', transform=transform) 
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=False, num_workers=4)

    # optimization 
    def phase_wrap(angle):
        return torch.remainder(angle, 2 * torch.pi)

    r2 = torch.linspace(0, args.MS_D, args.MS_D).to(device) * args.pitch

    loss_lpips = lpips.LPIPS(net='vgg').to(device) # B x C x H x W 
    loss_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # 1.0은 이미지가 [0,1] 범위일 때
    


    phase_init = torch.from_numpy(np.load('./data/ours_phase.npy')).to(device) #100.npy
    phase_init = torch.clamp(phase_init.unsqueeze(0), 0, torch.pi)

    phase_map = phase_init

    proxy = ProxyModel(args, device)  
    fdtd_proxy = FDTD_interp(device)

    # adaptive sampler for r 
    psf_simulator = psfSimulator(args.R, args.C, args.MS_D, args.R_cam, args.C_cam, args.pitch, args.pitch_light, args.pitch_cam, device, 'SBL_ASM')
    imgSim = ImgSimulator(args, device)
    
    num_angles = np.full(args.N_sample,10) 
    num_angles[0] = 1 
    v_theta = torch.linspace(0, 50, args.N_sample).to(device) 
    wvl_sample = torch.Tensor([850e-9]).to(device)
    wvl_sample = wvl_sample[:, None, None, None] 
    
    yy, xx = torch.meshgrid(
    torch.linspace(-1, 1, 600).to(device),
    torch.linspace(-1, 1, 600).to(device),
    indexing='ij'
        )
    rr_512 = torch.sqrt(xx**2 + yy**2)
    mask = rr_512 < 0.9
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    
    
    psnr_arr = 0 
    lpips_arr = 0 
    ssim_arr = 0 
    idxx = 0 

    for idx_batch, data in enumerate(dataloader_train): #train):
        img_ring = data.to(device)
        num_psfs = v_theta.shape[0]
        fe_list= get_psfs(args, phase_map, v_theta, proxy, fdtd_proxy)
        #fe_list= 1
        
        print("focusing efficiency: ", fe_list * 100)
 