import torch 
import torch.nn as nn 
import pickle 
import scipy.io
import numpy as np  
import os

class Phase2duty(): 
    def __init__(self, conf_file, device): 
        with open(conf_file, "r") as f:
            degree = int(f.readline())
            coefficients = list(map(float, f.readline().split()))
            
        self.degree = degree 
        self.coefficients = torch.Tensor(coefficients).to(device)
        
    def forward(self, phase): 
        phase_norm = (phase % (2*torch.pi)) / (2*torch.pi) # 0 ~ 1 
        y = sum([self.coefficients[i] * (phase_norm ** (self.degree-i)) for i in range(len(self.coefficients))])
        
        return y
       
class FDTDModel:
    def __init__(self, coef_file, device): 
        
        with open(coef_file, 'rb') as f: 
            data = pickle.load(f)
            
        self.TM = data['TM']
        self.TE = data['TE']
        
    
    # d: duty cycle [0 , 1], a : angle of incident light [0, 1] (0 ~ 50 degree)
    def duty2data(self, d, a, mode='TM'):
        if mode=='TM':
            b, A_coef, B_coef, d0_coef = self.TM 
        else: 
            b, A_coef, B_coef, d0_coef = self.TE 
            
        b0, b1, b2, b3, b4, b5 = b 
        poly = b0*d**2 + b1*a**2 + b2*d*a + b3*d + b4*a + b5 
        
        A = A_coef[0] + A_coef[1]*a + A_coef[2]*a**2 
        B = B_coef[0] + B_coef[1]*a + B_coef[2]*a**2 
        d0 = d0_coef[0] + d0_coef[1]*a + d0_coef[2]*a**2 
        
        ridge = A * torch.atan(B * (d-d0))
        
        result = poly + ridge 
        
        return result 
    
    def forward(self, d, a): 
        TM_result = self.duty2data(d, a, 'TM')
        TE_result = self.duty2data(d, a, 'TE')
        
        return TM_result, TE_result 
        
        
class ProxyInterp:
    def __init__(self, args, device):
        self.device = device
        self.args = args   
        self.ph2duty = Phase2duty(args.phase2struct, device)
        # T 
        self.p_TM = torch.from_numpy((scipy.io.loadmat(os.path.join(args.fdtd_path, 'phaseEx(TM).mat'))['phaseEx'])).to(device)
        self.p_TE = torch.from_numpy((scipy.io.loadmat(os.path.join(args.fdtd_path, './phaseEy(TE).mat'))['phaseEy'])).to(device)
        self.t_TM = torch.from_numpy((scipy.io.loadmat(os.path.join(args.fdtd_path, './Tx(TM).mat'))['Tx'])).to(device)
        self.t_TE = torch.from_numpy((scipy.io.loadmat(os.path.join(args.fdtd_path, './Ty(TE).mat'))['Ty'])).to(device)
        
        
    def bilinear_interp(self, z_grid, d_query, a_query): 
        # z_grid : 26 x 21 
        # d_query, a_query: (M,) torch.tensors in [0, 1]
        # returns: (M, interpolated values)
        
        H, W = z_grid.shape 
        d_idx = d_query * (H-1)
        a_idx = a_query * (W-1)
        
        i0 = d_idx.floor().long().clamp(0, H - 2)
        i1 = i0 + 1
        j0 = a_idx.floor().long().clamp(0, W - 2)
        j1 = j0 + 1

        wd = d_idx - i0.float()
        wa = a_idx - j0.float()

        # Gather 4 neighbors
        z00 = z_grid[i0, j0]
        z01 = z_grid[i0, j1]
        z10 = z_grid[i1, j0]
        z11 = z_grid[i1, j1]

        # Bilinear interpolation
        z_interp = (
            (1 - wd) * (1 - wa) * z00 +
            (1 - wd) * wa       * z01 +
            wd       * (1 - wa) * z10 +
            wd       * wa       * z11
        )
        return z_interp
        
    def forward(self, phasemap, aoi):
        diameters = self.ph2duty.forward(phasemap)
        T_tm = self.bilinear_interp(self.t_TM, diameters, torch.full_like(diameters, aoi))
        T_te = self.bilinear_interp(self.t_TE, diameters, torch.full_like(diameters, aoi))
        P_tm = self.bilinear_interp(self.p_TM, diameters, torch.full_like(diameters, aoi))
        P_te = self.bilinear_interp(self.p_TE, diameters, torch.full_like(diameters, aoi))
        
        A_tm, A_te = torch.sqrt(torch.clamp(T_tm, 0, 1.)), torch.sqrt(torch.clamp(T_te, 0, 1.))
        
        return [A_tm, P_tm, A_te, P_te ]
         
        
      
class ProxyModel():
    def __init__(self, args, device):
        self.device = device 
        self.args = args
        
        self.ph2duty = Phase2duty(args.phase2struct, device)
        self.duty2T = FDTDModel(args.proxy_T, device)
        self.duty2Phase = FDTDModel(args.proxy_phase, device)
        

            
    def forward(self, phasemap, aoi):
        diameters = self.ph2duty.forward(phasemap)
        
        T_tm, T_te = self.duty2T.forward(diameters, aoi)
        A_tm, A_te = torch.sqrt(torch.clamp(T_tm, 0, 1.)), torch.sqrt(torch.clamp(T_te, 0, 1.))
        
        P_tm, P_te = self.duty2Phase.forward(diameters, aoi)
        
        
        return [A_tm, P_tm, A_te, P_te] # result 
        
                