import torch
import random
import numpy as np
from dataset import *
import lpips 
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure



from torchvision.transforms.functional import gaussian_blur

from imgSimul import ImgSimulator
from torch.utils.data import  DataLoader 
from proxyModel import * 
from adapSampler import Sampler_R, angle_sampling
from psfSimul import psfSimulator
from ArgParser import Argument 
from utils_noise import GaussianNoise


import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg') 


def get_psfs(args, phase_map, v_theta, proxy):
    k = args.k 
    patches = []
    r_list = []
    fe_list = []
    fe_na_list = []
    psf_list = None
    
    for i in range(v_theta.shape[0]):
        angle = v_theta[i]
        y = -args.f * torch.sin(torch.deg2rad(v_theta[0]))
        x = torch.zeros_like(y)
        offset = [y, x]
        
        result_FDTD = proxy.forward(phase_map, angle/50.)
           
        psf, input_light = psf_simulator.forward2(result_FDTD, args.sensor_dist, offset, args.wvl, -1*angle, mode='far')
        if psf_list is None: 
            psf_list = psf 
        else: 
            psf_list += psf 
         

    psf_sensor =  F.interpolate(psf_list, scale_factor=args.pitch/args.pitch_cam, mode='bilinear')  # 1 x 1 x H x W 

    for i in range(v_theta.shape[0]):
        
        y = int(-args.f * torch.sin(torch.deg2rad(v_theta[i])) / args.pitch_cam)
        _, _, h, w = psf_sensor.shape
        
        ch, cw = h//2 + y, w//2 
        r = args.f * torch.sin(torch.deg2rad(v_theta[i])) / args.pitch_cam
        cropped_psf = psf_sensor[0][0][ch-k:ch+k, cw-k:cw+k]
        
        fe = cropped_psf.sum() / input_light.sum() 
        #
        
        # for focusing efficiency  
        #y2 = int(-args.f * torch.sin(torch.deg2rad(v_theta[i])) / 400e-9 )
        _, _, h, w = psf_list.shape 
        #ch, cw = h//2 + y2, w//2 
        
        NA = args.na 
        wavelength = args.wvl
        pixel_pitch = args.pitch
        airy_radius_m = 1.22 * wavelength / NA * 3 
        airy_radius_px = airy_radius_m / pixel_pitch
        ch = h//2 - int(args.f * torch.sin(torch.deg2rad(v_theta[i])) / args.pitch)
        cw = w // 2 
        
        y = np.arange(h).reshape(-1, 1)
        x = np.arange(w).reshape(1, -1)
        dist_sq = (x-cw)**2 + (y-ch)**2 
        mask = dist_sq <= airy_radius_px**2  
        
        focusing = psf_list[0][0][mask]
        focusing_efficiency = focusing.sum() / input_light.sum()
        
        fe_na_list.append(focusing_efficiency)        
        fe_list.append(fe.detach())
        patches.append(cropped_psf)
        r_list.append(r)
        
    patches = torch.stack(patches)
    fe_na_list = torch.stack(fe_na_list)
    r_list = torch.stack(r_list)
    fe_list = torch.stack(fe_list)
    
    return patches, r_list, fe_list, fe_na_list

        
 


if __name__ == "__main__":
    parser = Argument() 
    args = parser.parse()
    
    device = torch.device(args.device)
    ref_wvl = args.wvl
    
    loss_fn = lpips.LPIPS(net='vgg').to(device) # B x C x H x W 
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # 1.0은 이미지가 [0,1] 범위일 때

    
    # deblur network 
    # 'no', 'net', 'wiener'
    model = None 
    
    # tensorboard 
    writer = SummaryWriter(log_dir=args.log_path)
    
    # dataset
    dataset_train = MyDataset(args.train_dataset_path, 'train', transform=transform) 
    dataset_valid = MyDataset(args.valid_dataset_path, 'test', transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=4)

    # optimization 
    def phase_wrap(angle):
        return torch.remainder(angle, 2 * torch.pi)

    r2 = torch.linspace(0, args.MS_D, args.MS_D).to(device) * args.pitch
    
    phase_quad = (-torch.pi/850e-9) * r2**2 / args.f 
    phase_wrapped = (phase_quad) % (2 * torch.pi)
    phi_half = torch.where(phase_wrapped < torch.pi, phase_wrapped, 2 * torch.pi - phase_wrapped)
    init = (phi_half / torch.pi).clamp(1e-6, 1-1e-6)
    
    raw_param = torch.log(init / (1-init))
    raw_param = torch.nn.parameter.Parameter(data=raw_param.unsqueeze(0), requires_grad=True) 


    optimizer_meta = optim.Adam([raw_param], lr=args.lr_meta) if args.MS_train else None
    optimizer_net = optim.Adam(model.parameters(), lr=args.lr_net) if args.use_net else None
    scheduler_meta = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=700, gamma=0.5) if args.MS_train else None
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=600, gamma=0.1) if args.use_net else None

    # proxy model 
    proxy = ProxyModel(args, device) 
    proxy_interp = ProxyInterp(args, device)

    # image simulation
    sampler_r = Sampler_R(N_min=args.r_sample_min, N_max=args.r_sample_max, device=device, N_init=args.N_sample, bandwidth=5)
    psf_simulator = psfSimulator(args.R, args.C, args.MS_D, args.R_cam, args.C_cam, args.pitch, args.pitch_light, args.pitch_cam, device, 'SBL_ASM')
    imgSim = ImgSimulator(args, device)
    
    num_angles = np.full(args.N_sample,10) 
    num_angles[0] = 1 
    v_theta = torch.linspace(0, 50, args.N_sample).to(device) 
    wvl_sample = torch.Tensor([850e-9]).to(device)
    wvl_sample = wvl_sample[:, None, None, None] 
    
    losses_train, losses_val, psnrs_val = [], [], []
    Noise_func = GaussianNoise(0, 0.02 * torch.pi , device) 
    
    yy, xx = torch.meshgrid( torch.linspace(-1, 1, 600).to(device), torch.linspace(-1, 1, 600).to(device),indexing='ij')
    rr_512 = torch.sqrt(xx**2 + yy**2)
    mask = rr_512 < 0.9
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    threshold = 0.02
    
    use_interp = True
    epoch = 1000
    for e in range(epoch):
        # train
        optimizer_meta.zero_grad() if optimizer_meta is not None else None
        optimizer_net.zero_grad() if optimizer_net is not None else None 
        model.train() if optimizer_net is not None else None 
    
            
        loss_train = 0 
        fe = 0 
        for idx_batch, data in enumerate(dataloader_train):
            
            # 0~ pi constraint
            phase_map = torch.sigmoid(raw_param) * torch.pi / 2 + torch.pi/2  + Noise_func.sample(raw_param.shape)
            img_ring = data.to(device)
            num_psfs = v_theta.shape[0]

            if use_interp: 
                patches, r_list, fe_list, fe_na_list = get_psfs(args, phase_map, v_theta, proxy_interp) 
            else:
                patches, r_list, fe_list, fe_na_list = get_psfs(args, phase_map, v_theta, proxy) 
                       
            result_conv, _ = imgSim.ring_convolution2(img_ring.clone(), r_list, patches, num_angles)

            r_norm = r_list / (args.R_cam / 2)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, args.R_cam, device=device), torch.linspace(-1, 1, args.R_cam, device=device), indexing='ij')
            rr = torch.sqrt(xx**2 + yy**2) 
            scale_map = torch.zeros_like(rr)
            
            
            output = result_conv * 2 - 1#fe normalization for LPIPS 
            gt = img_ring * 2 -1 
            
            def softmin(x, beta=10):
                return -torch.logsumexp(-beta * x, dim=0) / beta
             

            mse = F.mse_loss(output*mask, gt*mask)
            lpips_loss = loss_fn(output.unsqueeze(1), gt.unsqueeze(1)).mean()
            min_psf = softmin(fe_na_list, 1e+4)
            
            loss =  mse*5 + lpips_loss -min_psf * 1e+2 
            
            loss.backward() 
            optimizer_meta.step() if optimizer_meta is not None else None
            optimizer_net.step() if optimizer_net is not None else None 
            loss_train += loss.item() 

        
        scheduler_meta.step() if scheduler_meta is not None else None
        scheduler_net.step() if scheduler_net is not None else None 
        loss_train = loss_train / (idx_batch+1)
        writer.add_scalar('Train loss', loss_train, e)
        losses_train.append(loss_train) 

        # validation
        model.eval() if optimizer_net is not None else None 
        loss_valid = 0 
        psnr_valid = 0 
        
        patches_detach = patches.detach().clone() 
        with torch.no_grad(): 
            for idx_val, val_data in enumerate(dataloader_valid):
                val_img_ring = val_data.to(device)
                conv_val, _ = imgSim.ring_convolution2(val_img_ring.clone(), r_list, patches_detach, num_angles)
        
                result_fe = conv_val * scale_map
                output = conv_val
                gt = val_img_ring
                mask = (rr < 1)
                
                #mse = F.mse_loss(output*mask, gt*mask)
                #l1loss = F.l1_loss(output * mask, gt * mask)
                #ssim_loss = 1 - ssim(output.unsqueeze(1), gt.unsqueeze(1))
                lpips_loss = loss_fn(output.unsqueeze(1), gt.unsqueeze(1)).mean()
                loss =  lpips_loss - softmin(fe_na_list, 1e+4) * 1e+2   #- softmin(fe_list) * 1e+1 #3e+1 # 3e+1 
                #loss = mse + lpips_loss * 0.1 - softmin(fe_list, 1e+4) * 3e+1 
                
                #loss = F.mse_loss(output * mask, gt * mask) +  torch.var(fe_list)*0.5 + torch.mean(1-fe_list) * 0.1 
                #loss  = loss_fn(output.unsqueeze(1), gt.unsqueeze(1)).mean() -softmin(fe_list, 1e+4) * 3e+2
                
                psnr = 10 * torch.log10(1/loss)
                loss_valid += loss.item() 
                psnr_valid += psnr.item() 
                
            loss_valid = loss_valid/(idx_val+1)
            psnr_valid = psnr_valid /(idx_val+1)
            writer.add_scalar('Valid loss', loss_valid, e)
            writer.add_scalar('PSNR', psnr_valid, e)
            v_theta[0] = 0   
                 
        
        
        print(f"Epoch: {e+1}, Train Loss: {loss_train}, Valid Loss: {loss_valid}") 

    np.save(os.path.join(args.log_path, 'phase.npy'), phase_map.detach().cpu().numpy())










