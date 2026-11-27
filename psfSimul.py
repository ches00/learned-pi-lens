import torch
import numpy as np
import pado
import pickle
import math

from pado.conv import *

import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F


def phase_wrapping(phase):
    return (phase + torch.pi) % (2*torch.pi) - torch.pi



class psfSimulator():
    def __init__(self, R, C, MS_D, R_cam, C_cam, pitch_meta,  pitch_light, pitch_cam, device, prop_model='SBL_ASM'):
        self.R, self.C = R, C
        self.device = device
        self.MS_D = MS_D 
        self.pitch = pitch_meta
        self.prop_model = prop_model
        self.pitch_light = pitch_light 
        self.pitch_cam = pitch_cam
        self.R_cam, self.C_cam = R_cam, C_cam 
        

        
        dy = torch.linspace(-R//2, R//2, R).to(device) #* pitch
        dx = torch.linspace(-C//2, C//2, C).to(device) #* pitch
        
        dy, dx = torch.meshgrid(dy, dx) #* pitch
        
        r = torch.sqrt(dx**2 + dy**2)
        r = r.t()
        
        self.idx = r.long() 
        
        self.valid_idx = (self.idx >= 0) & (self.idx < MS_D)
        
        
        
        
        
        
    def recon2D(self, data):
        data2d = torch.zeros((data.shape[0], 1, self.R, self.C)).to(self.device)
        data = data.float()
        
        for i in range(data.shape[0]):
            data2d[i][0][self.valid_idx] = data[i, self.idx[self.valid_idx]]
            
        return data2d 
        
        
        
        
    def forward2(self, MS_modul, prop_z, offset, wvl, incident_angle, mode='far'):

        A_tm, P_tm, A_te, P_te = MS_modul # TM and TE mode 
        N = A_tm.shape[0]
        
        Amp_tm = self.recon2D(A_tm)
        Phase_tm = self.recon2D(P_tm)
        Amp_te = self.recon2D(A_te)
        Phase_te = self.recon2D(P_te)
        
        meta_tm = pado.DOE([N, 1, self.R, self.C], self.pitch, 'phase', wvl, self.device)
        meta_te = pado.DOE([N, 1, self.R, self.C], self.pitch, 'phase', wvl, self.device)
        
        meta_tm.set_phase_change(Phase_tm, wvl)
        meta_tm.set_amplitude_change(Amp_tm)
        meta_te.set_phase_change(Phase_te, wvl)
        meta_te.set_amplitude_change(Amp_te)

        #light_tm = pado.Light([N, 1, self.R, self.C], self.pitch, wvl, None, self.device) 
        #light_te = pado.Light([N, 1, self.R, self.C], self.pitch, wvl, None, self.device)
        
        #if mode == 'far':
        #    light_tm.set_plane_light(incident_angle) # single wavelength
        #    light_te.set_plane_light(incident_angle) 
        
        prop = pado.Propagator(self.prop_model) # shifted BL ASM 
        #prop = pado.Propagator('BL_ASM')
        diameter = self.pitch * self.R 
        
        aperture = pado.Aperture([1, 1, self.R, self.C], self.pitch, diameter, 'circle', wvl, self.device)
        result_psf = 0
        result_fe = 0
        result_fe_na = 0
        
        for meta in [meta_tm, meta_te]:
            light = pado.Light([N, 1, self.R, self.C], self.pitch, wvl, None, self.device) 
            light.set_plane_light(incident_angle)
            
            input_light = aperture.forward(light.clone())
            after_meta = meta.forward(input_light.clone())
            psf_ms = prop.forward(after_meta, prop_z, offset, linear=True).get_intensity()
            #psf_ms = prop.forward(after_meta, prop_z,  linear=True).get_intensity()
            
            
            result_psf += psf_ms / 2. # TM + TE 
            
        #print(fe_list2 * 100)   
        #psf_cam = F.interpolate(result_psf.unsqueeze(0), scale_factor=self.pitch_light/self.pitch_cam, mode='bilinear') # 1 x 1x H x W 

        return result_psf.unsqueeze(0), input_light.get_intensity() #, input_light.get_intensity().sum() psf_cam, result_fe, result_fe_na
    
    def forward(self, phase_1d,  x, y, z,  prop_z, wvl_sample, wvl=850e-9, incident_angle=None, A=None, mode='far'):
        N = phase_1d.shape[0]

        if phase_1d.shape[-1] > 1000:
            meta = pado.DOE([N, 1, self.R, self.C], self.pitch, 'phase', wvl, self.device)
            meta.set_phase_change(phase_1d.unsqueeze(1).reshape(N, 1, self.R, self.C), wvl)
        else:
            phase_2d = torch.zeros((N, self.R, self.C)).to(self.device)
            for i in range(phase_1d.shape[0]):
                phase_2d[i][self.valid_idx] = phase_1d[i, self.idx[self.valid_idx]]

            meta = pado.DOE([N, 1, self.R, self.C], self.pitch, 'phase', wvl, self.device)
            meta.set_phase_change(phase_2d.unsqueeze(1), wvl)

        if A is not None: 
            meta.set_amplitude_change(A)

        padding = int(self.R_cam * (self.pitch_cam / self.pitch_light))
        if padding % 2 != 0:
            padding += 1 

        light = pado.Light([N, 1, padding, padding], self.pitch, wvl, None, self.device) #self.R, self.C], self.pitch, wvl, None, self.device) #self.R , self.C],self. pitch, wvl, None, self.device)
        
        if mode=='near':
            light.set_spherical_light(z, dx=x, dy=y, wvls=wvl_sample)
        elif mode=='far': 
            light.set_plane_light(incident_angle, wvls=wvl_sample)
        elif mode =='gt':
            light.set_xy_plane_light(incident_angle, wvls=wvl_sample)
        

        prop = pado.Propagator(self.prop_model)

        diameter = self.pitch * self.R 
        aperture = pado.Aperture([1, 1, self.R, self.C], self.pitch, diameter, 'circle', wvl, self.device)

        after_meta = meta.forward(aperture.forward(light.clone()))  
        psf_ms = prop.forward(after_meta, prop_z, None, wvl_sample, True).get_intensity() 
        
        input_light = aperture.forward(light.clone()).get_intensity().sum() 
        
        _, h, w = psf_ms.shape 
        c = (psf_ms == psf_ms.max()).nonzero(as_tuple=False)[0]
        ch, cw = c[1], w//2 
        
        k = 50
        cropped_psf = psf_ms[0][ ch-k:ch+k, cw-k:cw+k]
        
        
        NA = 0.78
        center = k #40
        wavelenght = 850e-9 
        pixel_pitch = 420e-9 
        airy_radius_m = 1.22 * wavelenght / NA * 3 
        airy_radius_px = airy_radius_m / pixel_pitch 
        
        y, x = np.indices((2*k, 2*k))
        distances = np.sqrt((x-center)**2 + (y-center)**2)
        mask = distances <= airy_radius_px
        intensity_sum = torch.sum(cropped_psf[mask])
        fe = intensity_sum / input_light
        
        #psf_ms_norm = psf_ms / input_light 
        #scale = psf_ms_norm.sum() 
        psf_cam = F.interpolate(psf_ms.unsqueeze(1), scale_factor=self.pitch_light/self.pitch_cam, mode='bilinear') # interpolation 
        #result = (psf_cam / psf_cam.sum()) * scale

        return psf_cam, fe#, scale #, aperture.forward(light.clone()).get_intensity().sum().detach() #psf_ms.unsqueeze(1), aperture.forward(light.clone()).get_intensity() 

