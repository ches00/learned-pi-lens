import torch 
from torch.nn import functional as F
import numpy as np

def shift_similarity(K1, K2):
    max_correlation = -float('inf')
    best_shift = (0, 0)
    shift_range = 8 

    K1 = K1[None, None, :, :]
    K2 = K2[None, None, :, :]

    #K1_norm = torch.norm(K1)  # 고정된 값을 반복적으로 계산하지 않도록 사전 계산

    # 패딩 추가
    padded_kernel1 = F.pad(K1, (shift_range, shift_range, shift_range, shift_range))
    padded_kernel2 = F.pad(K2, (shift_range, shift_range, shift_range, shift_range))
    K1_norm = torch.norm(padded_kernel1)

    for dx in range(-shift_range, shift_range + 1):
        for dy in range(-shift_range, shift_range + 1):
            # torch.roll로 시프트 적용
            shifted_kernel2 = torch.roll(padded_kernel2, shifts=(dx, dy), dims=(-2, -1))

            # 유사도 계산
            correlation = (padded_kernel1 * shifted_kernel2).sum() / (K1_norm * torch.norm(shifted_kernel2))

            if correlation > max_correlation:
                max_correlation = correlation
                best_shift = (dx, dy)
                
    return max_correlation, best_shift#, shifted_kernel2 



class Sampler_R:
    def __init__(self, N_min, N_max, device, N_init=10, bandwidth=3):
        self.bandwidth=bandwidth 
        self.N_init = N_init 
        self.area_init = 0 
        self.device = device
        
        self.N_min = N_min 
        self.N_max = N_max
        
        self.x_grid = torch.linspace(0, 50, 1000).to(self.device)
        self.x_grid = self.x_grid.unsqueeze(0)
        
    def psf_sampling(self, X, patches):
        S = []
        offset = []
        for i in range(X.shape[0]-1):
            K1 = patches[i]
            K2 = patches[i+1]
            
            s, best_shift  = shift_similarity(K1, K2)
            offset.append(best_shift)
            S.append(s)
        SS = torch.stack(S)
        return SS, offset
    
    def recon_pdf(self, X, similarity):
        x_mid = (X[1:]+X[:-1])/2 
        diff = x_mid.unsqueeze(1) - self.x_grid 
        
        grad = -torch.log(similarity)/x_mid 
        value = x_mid.sum() / x_mid.shape[0]
        #print(value)
        sampled = int((value - 18) /9 * 6 + 3) 
        
        kernels = torch.exp(-0.5 * (diff/self.bandwidth)**2) / (self.bandwidth * np.sqrt(2*np.pi)) * (-torch.log(similarity)).unsqueeze(1)
        pdf = kernels.sum(dim=0)
        
        return pdf, sampled  
    
    def inverse_transform_sampling(self, pdf, sampled):
        cumulative_integral = torch.cumsum(pdf, dim=0) * (self.x_grid[0][1] - self.x_grid[0][0])
        area = cumulative_integral[-1].item() 
        
        if self.area_init == 0:
            self.area_init = area 
        
        
        num_samples = self.N_init #int(self.N_init * np.sqrt(area/self.area_init) * 0.75) 
        #num_samples = np.clip(num_samples, self.N_min, self.N_max)
        
        total_integral = cumulative_integral[-1].item()
        interval_value = total_integral / (num_samples - 1)
        threshold_values = torch.linspace(0, total_integral, num_samples).to(self.device)
        
        indices = torch.searchsorted(cumulative_integral, threshold_values)
        indices = torch.clamp(indices, min=1, max=len(cumulative_integral)-1)
        
        uniform_samples = self.x_grid[0][indices]
        return uniform_samples, cumulative_integral[-1].item() 
    
    def sampling(self, v_theta, patches):
        SS, best_shift = self.psf_sampling(v_theta, patches)
        pdf, sampled = self.recon_pdf(v_theta, SS) 
        pdf = pdf / v_theta.shape[0]
        uniform_samples, _ = self.inverse_transform_sampling(pdf, sampled)
        
        return uniform_samples, _



def rotational_symmetry(psf):
    N = psf.shape[0]
    center = (N // 2, N // 2)

    # PSF 표준화
    psf_standardized =  (psf - psf.mean()) / psf.std()

    # 1. Polar Coordinate Transformation
    y, x = torch.meshgrid(torch.arange(N) - center[0], torch.arange(N) - center[1], indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    r_flat = r.view(-1)
 
    psf_flat = psf_standardized.view(-1)

    radial_bins = torch.linspace(0, r.max(), steps=20)
    #print(radial_bins)
    weighted_symmetry_measure = 0

    
    for i in range(1, len(radial_bins)):
        mask = (r_flat >= radial_bins[i-1]) & (r_flat < radial_bins[i])
        if mask.sum() > 0:
            values = psf_flat[mask]
            angular_variance = torch.std(values)
            #arr.append(angular_variance.item())
            weighted_symmetry_measure += radial_bins[i] * angular_variance
            #print( radial_bins[i])
    return weighted_symmetry_measure / radial_bins.sum() #symmetry_score
    

def angle_sampling(patches, num_psfs):
    result = []
    for i in range(num_psfs):
        #corr = rotational_symmetry(patches[i]) ** 3 / 6000  #(1) theta_only일 떄 
        corr = rotational_symmetry(patches[i])
        result.append(corr)
        #corr = (rotational_symmetry(patches[i])**2 ) * 80  #- 0.51) * 33 # **3 / 1.3e+4
        #print(corr) 
        #result.append(torch.clamp(corr, min=1, max=8).item() )
        
    result = torch.Tensor(result).to(patches.device)
    result = (result **2) * 76#80 
    num_angles = torch.clamp(torch.ceil(result), min=1, max=10).int()
    return num_angles

def angle_sampling_max(patches, num_psfs, max_n_angle):
    result = []
    for i in range(num_psfs):
        #corr = rotational_symmetry(patches[i]) ** 3 / 6000  #(1) theta_only일 떄 
        corr = rotational_symmetry(patches[i])
        result.append(corr)
        #corr = (rotational_symmetry(patches[i])**2 ) * 80  #- 0.51) * 33 # **3 / 1.3e+4
        #print(corr) 
        #result.append(torch.clamp(corr, min=1, max=8).item() )
        
    result = torch.Tensor(result).to(patches.device)
    result = (result **2) * 76#80 
    num_angles = torch.clamp(torch.ceil(result), min=1, max=max_n_angle).int()
    return num_angles









'''
# covariance analysis 

def calc_corr(target):

    image_flat = target.view(-1)

    weighted_mean_x = torch.sum(x_coords_flat * image_flat) / torch.sum(image_flat)
    weighted_mean_y = torch.sum(y_coords_flat * image_flat) / torch.sum(image_flat)
    weighted_mean_intensity = torch.mean(image_flat)

    x_diff = x_coords_flat - weighted_mean_x 
    y_diff = y_coords_flat - weighted_mean_y

    covv = (x_diff * y_diff * image_flat).sum() / torch.sum(image_flat)
    
    std_x = torch.sqrt((x_diff**2 * image_flat).sum() / torch.sum(image_flat))
    std_y = torch.sqrt((y_diff**2 * image_flat).sum() / torch.sum(image_flat))
    
    corr = covv / (std_x * std_y)
    
    return corr


def angle_sampling(patches, num_psfs):
    result = []
    for i in range(num_psfs):
        corr = calc_corr(patches[i]**3)
        result.append(corr.item())
    result = torch.Tensor(result).to(device)
    num_angles = torch.ceil(torch.abs(result)*24).int()
    return num_angles

def calculate_centroid(image_, threshold=None):
    
    if threshold:
        image = torch.nn.functional.threshold(image_, threshold, 0)
    else:
        image = image_
    total_intensity = torch.sum(image)
    
    yy, xx = torch.meshgrid(torch.arange(image.shape[-2]), torch.arange(image.shape[-1]))
    xx = xx.to(device)
    yy = yy.to(device)
    
    x_weighted_sum = torch.sum(xx * image)
    y_weighted_sum = torch.sum(yy * image)
    
    centroid_x = x_weighted_sum / total_intensity
    centroid_y = y_weighted_sum / total_intensity
    
    return centroid_y.long(), centroid_x.long()

def robust_centroid(psf, threshold_ratio=0.5):
    psf_detached = psf.detach()#**2
    threshold = psf_detached.max() * threshold_ratio
    indices = torch.nonzero(psf_detached >= threshold, as_tuple=True)
    data = psf_detached[indices[0], indices[1]]#**3 
    total_intensity = data.sum() 
    h_weighted_sum = torch.sum(indices[0] * data)
    w_weighted_sum = torch.sum(indices[1] * data)
    centroid_h = (h_weighted_sum / total_intensity).long()
    centroid_w = (w_weighted_sum / total_intensity).long()
    
    return centroid_h, centroid_w 


'''
        
        