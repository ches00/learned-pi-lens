from torch.nn.functional import pad 
import torch
from torch.nn import functional as F
from pado.conv import conv_fft_img_psf
from utils_noise import GaussianNoise
import numpy as np 


def get_rotation_matrix(angle):
    angle_rad = angle * torch.pi / 180 
    return [
        [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
        [torch.sin(angle_rad), torch.cos(angle_rad), 0]
    ]

class ImgSimulator:
    def __init__(self, args, device):
        self.R_cam = args.R_cam 
        self.C_cam = args.C_cam 
        self.k = args.k 
        self.num_angles = None #num_angles 
        self.device = device 
        
        self.noise = GaussianNoise(0, args.noise_gaussian_stddev, self.device)
        
        x = torch.linspace(0, 1, self.R_cam).to(device) - 0.5 
        y = torch.linspace(0, 1, self.C_cam).to(device) - 0.5 
        
        yy, xx = torch.meshgrid(y, x)
        r_img = torch.sqrt(xx**2 + yy**2)
        theta_img = torch.atan2(xx, yy) + torch.pi 
        theta_img = torch.rad2deg(theta_img)
        
        self.r_img = r_img 
        self.theta_img = theta_img 
        self.r_list = None #r_list 

        
    def set_r_list(self, r_list):
        self.r_list = r_list 
    
    def set_patches(self, patches):
        self.patches = patches 
        
    def annular_partitioning(self):
        rr = torch.Tensor(self.r_list).to(self.device)
        rr = rr / self.R_cam # 0 to 0.5 
        area = (rr[1:] + rr[:-1])/2
        area = torch.cat((torch.tensor([0.]).to(self.device), area, torch.tensor([1.0]).to(self.device)))

        return area
    
    def create_weighted_map(self, area, i, sigma=1.0):
        # smooth transition using a Gaussian-like weight for seamless convolution
        transition_zone = (area[i+1]-area[i]) / 2
        center =(area[i+1]+area[i]) / 2 
        weight = torch.exp(-0.5 * ((self.r_img-center) / transition_zone)**2)
        weight[self.r_img < area[i]] = 0.0
        weight[self.r_img >= area[i+1]] = 0.0 

        return weight
    
    
    def get_rotation_psf(self, patches):
        result = []

        for i, angle in enumerate(self.num_angles):
            angles = torch.linspace(0, 360, angle + 1)[:-1]

            theta = torch.tensor([get_rotation_matrix(angle) for angle in angles], dtype=torch.float32).to(self.device) 
            kernels_expanded = patches[i].expand(angle, 1,  -1, -1) # (N, 1, 76, 76)
            grids = F.affine_grid(theta, kernels_expanded.size(), align_corners=False)
            rotated_kernel = F.grid_sample(kernels_expanded, grids, align_corners=False)

            result.append(rotated_kernel)

        return result 

    def compute_window(self, img_batch):
        k = self.k 
        img_pad2 = pad(img_batch, (self.k, self.k, self.k, self.k), mode='replicate')# B x H x W 

        B, ih, iw = img_pad2.shape 
        arr_strides = img_pad2.stride() 

        out_shape = (B, ih-k*2, iw-k*2, k*2, k*2)
        out_strides = (arr_strides[0], arr_strides[1], arr_strides[2], arr_strides[1], arr_strides[2])

        windoes = torch.as_strided(img_pad2, size=out_shape, stride=out_strides)
        return windoes
    
    def gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(kernel_size).to(self.device) - (kernel_size - 1) / 2
        gauss = torch.exp(-0.5 * (x/sigma) ** 2)
        kernel_1d = gauss / gauss.sum() 
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d

    def create_gaussian_kernel(self, kernel_size=5, sigma=1):
        x = torch.arange(-kernel_size//2+1, kernel_size//2+1).to(self.device)
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t() 

        gaussian_kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum() 

        return gaussian_kernel.unsqueeze(0).unsqueeze(0)

    def ring_convolution2(self, img_ring, r_list, patches, num_angles):
        blend_size = blend_theta=20
        gauss_size = 5
        blend_r = 0.05
        sigma = 5
        offset = 25
        overlap_rho = 0.02 #0.01 #0.015 #0.01 #0.06  #0.06-ours #0.05#0.05 #0.02
        overlap_phi = 10 #3 #17 #25 #15 #15 #15 #10 # 5-degree overlap
        
        self.r_list = r_list 
        #self.patches = patches 
        self.num_angles = num_angles
        b, h, w = img_ring.shape
        
        rotated_psf_kernels = self.get_rotation_psf(patches)
        area = self.annular_partitioning()
        angle_list = []
        #windoes = self.compute_window(img_ring)
        result_ring = torch.zeros_like(img_ring)
        weight_map = torch.zeros((h, w)).to(self.device)


        for i in range(area.shape[0]-1):
            n_angle = self.num_angles[i]
            angles = torch.linspace(0, 360, n_angle + 1)#[:-1]
            area_angle = (angles[1:]+angles[:-1])/2
            area_angle = torch.cat([area_angle[n_angle-1:n_angle], area_angle])
            angle_list.append(angles)

            rho_min = area[i] - overlap_rho
            rho_max = area[i+1] + overlap_rho
            
            cond1 = torch.logical_and(self.r_img>=area[i], self.r_img < area[i+1])
            w_cond1 = torch.logical_and(self.r_img >= rho_min, self.r_img < rho_max)


            for j in range(area_angle.shape[0]-1):
                #overlap_phi = torch.clip(360//n_angle, max=10, min=0.5)
                theta_min = area_angle[j] - overlap_phi
                theta_max = area_angle[j+1] + overlap_phi
                img_ring2 = torch.zeros_like(img_ring)
                
                if j == 0:
                    cond2 = torch.logical_or(self.theta_img>=area_angle[j], self.theta_img < area_angle[j+1] )
                    w_cond2 = torch.logical_or(self.theta_img >= theta_min, self.theta_img < theta_max)
                else:    
                    cond2 = torch.logical_and(self.theta_img>=area_angle[j], self.theta_img < area_angle[j+1])
                    w_cond2 = torch.logical_and(self.theta_img>= theta_min, self.theta_img < theta_max)

                cond = torch.logical_and(cond1, cond2)
                w_cond = torch.logical_and(w_cond1, w_cond2)#.float() # binary weight map 
                h_indices, w_indices = torch.nonzero(w_cond, as_tuple=True)
                
                #h, w min max calculation 
                h_min, h_max = max(h_indices.min().item()-offset, 0) , min(h_indices.max().item() +offset, h)
                w_min, w_max = max(w_indices.min().item()-offset, 0),  min(w_indices.max().item() +offset, w) 
                
                #print(h_min, h_max, w_min, w_max)
                # rectangle 
                ret = img_ring.clone()[:, h_min:h_max, w_min:w_max].unsqueeze(0) # 1 x B x h x w 
                psf_kernel = rotated_psf_kernels[i][j].unsqueeze(0)
                psf_kernel = psf_kernel / psf_kernel.sum()
                
                ret2 = conv_fft_img_psf(ret, psf_kernel).squeeze(0) # B x h x w
                result = ret2[:, w_cond[h_min:h_max, w_min:w_max]]
                
                # weight
                #gauss_weight = torch.ones_like(weight_map)
                kernel_size = 5#15 #15
                kernel = self.gaussian_kernel(kernel_size, 3) # 9)
                kernel = kernel.expand(1, 1, kernel_size, kernel_size)
                
                gauss_weight = F.conv2d(cond.float().unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2)

                
                img_ring2[:, w_cond] = result 
                result_ring += img_ring2 * gauss_weight.squeeze()

                weight_map += gauss_weight.squeeze()
                

        result_ring = result_ring / (weight_map + 1e-8) #img_ring2 #result_ring / weight_sum   
        noise = self.noise.sample(result_ring.shape)
        im_sim_noisy = result_ring + noise 
        im_sim_noisy_clamped = torch.clamp(im_sim_noisy, min=0, max=1)

        return im_sim_noisy_clamped, rotated_psf_kernels

