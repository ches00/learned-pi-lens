import torch
import numpy as np
from .math import fft, ifft, conv_fft, wrap_phase
import torch.nn.functional as F


def compute_pad_width(field, linear):
    """
    Compute the pad width of an array for FFT-based convolution
    Args:
        field: (B,Ch,R,C) complex tensor
        linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
    Returns:
        pad_width: pad-width tensor
    """

    if linear:
        R,C = field.shape[-2:]
        pad_width = (C//2, C//2, R//2, R//2)
    else:
        pad_width = (0,0,0,0)
    return pad_width 

def unpad(field_padded, pad_width):
    """
    Unpad the already-padded complex tensor 
    Args:
        field_padded: (B,Ch,R,C) padded complex tensor 
        pad_width: pad-width tensor
    Returns:
        field: unpadded complex tensor
    """

    field = field_padded[...,pad_width[2]:-pad_width[3],pad_width[0]:-pad_width[1]]
    return field

class Propagator:
    def __init__(self, mode, polar='non'):
        """
        Free-space propagator of light waves
        One can simulate the propagation of light waves on free space (no medium change at all).
        Args:
            mode: type of propagator. Currently, we support "Fraunhofer", "Fresnel", "ASM" propagation methods.
        """
        self.mode = mode
        self.polar = polar

    def forward(self, light, z, offset=None, wvl_sample=None, linear=True, target_plane=None):
        """
        Forward the incident light with the propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """
        
        if z == 0:
            # No propagation
            return light
    
        if self.polar=='non':
            return self.forwardNonPolar(light, z, offset, wvl_sample, linear, target_plane)
        elif self.polar=='polar':
            x = self.forwardNonPolar(light.get_lightX(), z, offset, linear)
            y = self.forwardNonPolar(light.get_lightY(), z, offset, linear)
            light.set_lightX(x)
            light.set_lightY(y)
            return light
        else:
            NotImplementedError('Polar is not set.')

    def forwardNonPolar(self, light, z, offset=None, wvl_sample=None, linear=True, target_plane=None):
        """
        Forward the incident light with the propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        if self.mode == 'Fraunhofer':
            return self.forward_Fraunhofer(light, z, linear)
        if self.mode == 'Fresnel':
            return self.forward_Fresnel(light, z, linear)
        if self.mode == 'ASM':
            return self.forward_ASM(light, z, linear)
        if self.mode == 'BL_ASM':
            return self.forward_BL_ASM(light, z, linear)
        if self.mode == 'fixed_ASM':
            return self.forward_fixed_ASM(light, z, linear)
        if self.mode == 'fixed_Fresnel':
            return self.forward_fixed_Fresnel(light, z, linear)
        if self.mode == 'SBL_ASM':
            return self.forward_shifted_BL_ASM(light, z, offset, linear)
        if self.mode == 'RS':
            return self.forward_RayleighSommerfeld(light, z, linear, target_plane)
        else:
            return NotImplementedError('%s propagator is not implemented'%self.mode)


    def forward_Fraunhofer(self, light, z, linear=True):
        """
        Forward the incident light with the Fraunhofer propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
                The propagated wavefront is independent w.r.t. the travel distance z.
                The distance z only affects the size of the "pixel", effectively adjusting the entire image size.
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        pad_width = compute_pad_width(light.field, linear)
        field_propagated = fft(light.field, pad_width=pad_width)
        field_propagated = unpad(field_propagated, pad_width)

        # based on the Fraunhofer reparametrization (u=x/wvl*z) and the Fourier frequency sampling (1/bandwidth)
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        pitch_r_after_propagation = light.wvl*z/bw_r
        pitch_c_after_propagation = light.wvl*z/bw_c

        light_propagated = light.clone()

        # match the x-y pixel pitch using resampling
        if pitch_r_after_propagation >= pitch_c_after_propagation:
            scale_c = 1
            scale_r = pitch_r_after_propagation/pitch_c_after_propagation
            pitch_after_propagation = pitch_c_after_propagation
        elif pitch_r_after_propagation < pitch_c_after_propagation:
            scale_r = 1
            scale_c = pitch_c_after_propagation/pitch_r_after_propagation
            pitch_after_propagation = pitch_r_after_propagation

        light_propagated.set_field(field_propagated)
        light_propagated.magnify((scale_r,scale_c))
        light_propagated.set_pitch(pitch_after_propagation)

        return light_propagated

    def forward_Fresnel(self, light, z, linear):
        """
        Forward the incident light with the Fresnel propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """
        field_input = light.field

        # compute the convolutional kernel 
        sx = light.dim[3] / 2
        sy = light.dim[2] / 2
        x = np.arange(-sx, sx, 1)
        y = np.arange(-sy, sy, 1)
        xx, yy = np.meshgrid(x,y)
        xx = torch.from_numpy(xx*light.pitch).to(light.device)
        yy = torch.from_numpy(yy*light.pitch).to(light.device)
        k = 2*np.pi/light.wvl  # wavenumber
        phase_u = (k*(xx**2 + yy**2)/(2*z))
        phase_u = phase_u.unsqueeze(0).unsqueeze(0)
        phase_w = wrap_phase(phase_u, stay_positive=False)
        amplitude = torch.ones_like(phase_w) / z / light.wvl
        conv_kernel = amplitude * torch.exp(phase_w*1j)
        conv_kernel /= conv_kernel.abs().sum() 
        
        # Propagation with the convolution kernel
        pad_width = compute_pad_width(field_input, linear)
        field_propagated = conv_fft(field_input, conv_kernel, pad_width)

        # return the propagated light
        light_propagated = light.clone()
        light_propagated.set_field(field_propagated)

        return light_propagated

    def forward_fixed_Fresnel(self, light, z, linear):
        field_input = light.field
        light_propagated = light.clone()  # Assuming this method properly duplicates the Light object

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear) 

            # Adjust spatial domain calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                sx = light.dim[3]
                sy = light.dim[2]
                x = np.arange(-sx, sx, 1)
                y = np.arange(-sy, sy, 1)
            else:
                sx = light.dim[3] / 2
                sy = light.dim[2] / 2
                x = np.arange(-sx, sx, 1)
                y = np.arange(-sy, sy, 1)

            xx, yy = np.meshgrid(x,y)
            xx = torch.from_numpy(xx*light.pitch).to(light.device)
            yy = torch.from_numpy(yy*light.pitch).to(light.device)
            wvl = wavelengths[chan]

            k = 2*np.pi/wvl # wavenumber
            phase_u = (k*(xx**2 + yy**2)/(2*z))
            phase_u = phase_u.unsqueeze(0).unsqueeze(0)
            phase_w = wrap_phase(phase_u, stay_positive=False)
            amplitude = torch.ones_like(phase_w) / z / wvl
            conv_kernel = amplitude * torch.exp(phase_w*1j)
            conv_kernel /= conv_kernel.abs().sum()

            H = fft(conv_kernel)
            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H
            field_propagated = ifft(G, pad_width=pad_width)

            # return the propagated light
            light_propagated.set_field(field_propagated)

        return light_propagated

    def forward_shifted_BL_ASM(self, light, z, offset,linear):
        '''
        Forward the incident light with the angular spectrum method, supporting multi-wavelength channels.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset: offset (y, x) between source and target plane in meters for off-axis propagation
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''
        field_input = light.field
        light_propagated = light.clone()  # Clone the light object for the propagated light

        # Check if wvl is a list or a single float, and adjust accordingly
        #if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
        #    wavelengths = light.wvl
        #else:
        #    wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel
        
        # wvls [N, 1, 1, 1]


        
        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')
            fxx = fxx.unsqueeze(0)#.unsqueeze(0)
            fyy = fyy.unsqueeze(0)#.unsqueeze(0)
            #wvl = wavelengths[chan]
            k = 2 * torch.pi / light.wvl
            gamma = torch.sqrt(torch.abs(1. - (light.wvl*fxx)**2 - (light.wvl*fyy)**2))
            

            """
            use the following diffraction angle and spatial frequency relationship: sin(theta_x)/wvl = f_x
            Below theta_limit is derived from geometry. When Bandwidth = 1/2/pitch, freq limit from Nyquist limit and geometry is same, which is the case
            when we use linear convolution to avoid circular convolution.
            Refer to Matsushima, "Shifted angular spectrum method for off-axis numerical propagation"
            """
            
            # define constants to compute bandlimit 
            bw_x = light.dim[-1]*light.pitch 
            bw_y = light.dim[-2]*light.pitch 
            u_limit_plus = 1/torch.sqrt(1+torch.pow(z/(offset[-1]+bw_x+1e-7),2))/light.wvl 
            u_limit_minus = 1/torch.sqrt(1+torch.pow(z/(offset[-1]-bw_x+1e-7),2))/light.wvl 
            v_limit_plus = 1/torch.sqrt(1+torch.pow(z/(offset[-1]+bw_y+1e-7),2))/light.wvl 
            v_limit_minus = 1/torch.sqrt(1+torch.pow(z/(offset[-1]+bw_y+1e-7),2))/light.wvl 
            
            '''
            if offset is not None:
                fx_max = (torch.sin(torch.arctan((offset[-1] + light.dim[-1]*light.pitch) / abs(z)))  + np.radians(0.1)) / light.wvl
                fx_min = (torch.sin(torch.arctan((offset[-1] - light.dim[-1]*light.pitch) / abs(z)))  + np.radians(0.1)) / light.wvl
                fy_max = (torch.sin(torch.arctan((offset[-2] + light.dim[-2]*light.pitch) / abs(z)))  + np.radians(0.1)) / light.wvl
                fy_min = (torch.sin(torch.arctan((offset[-2] - light.dim[-2]*light.pitch) / abs(z)))  + np.radians(0.1)) / light.wvl
                uv_filter = (fx_min < fxx) & (fx_max > fxx) & (fy_min < fyy) & (fy_max > fyy)
                gamma_offset = light.wvl*fxx*offset[1] + light.wvl*fyy*offset[0]
                H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * uv_filter).to(light.device)
            else:
                fxx_limit = torch.sin(torch.arctan(light.dim[-1]*light.pitch/abs(z)))/light.wvl
                fyy_limit = torch.sin(torch.arctan(light.dim[-2]*light.pitch/abs(z)))/light.wvl
                uv_filter = (torch.abs(fxx) < fxx_limit) & (torch.abs(fyy) < fyy_limit)
                H = (torch.exp(1j*k*z*gamma) * uv_filter).to(light.device)
            '''
            wavelengths = light.wvl 
            
            if offset[-1] > bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_plus,2))
                fxx_lower_bound = fxx >= torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_minus,2))
            elif offset[-1] <= bw_x and offset[-1] >= -bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_plus,2)) - (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-1]==-bw_x else 0)
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_minus,2)) + (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-1]==bw_x else 0)
            else:
                fxx_upper_bound = fxx <= -torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_plus,2))
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_minus,2))

            # fyy bound
            if offset[-2] > bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_plus,2))
                fyy_lower_bound = fyy >= torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_minus,2))
            elif offset[-2] <= bw_y and offset[-2] >= -bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_plus,2)) - (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-2]==-bw_y else 0)
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_minus,2)) + (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-2]==bw_y else 0)
            else:
                fyy_upper_bound = fyy <= -torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_plus,2))
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_minus,2))

            uv_filter = fxx_upper_bound & fxx_lower_bound & fyy_upper_bound & fyy_lower_bound      
            
            gamma_offset = light.wvl*fxx*offset[1] + light.wvl*fyy*offset[0]
            H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * uv_filter).to(light.device)      

            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H

            field_propagated = ifft(G, pad_width=pad_width)

            # Set the propagated field for each channel
            light_propagated.set_field(field_propagated)

        return light_propagated
    
    def forward_ASM(self, light, z, linear):
        '''
        Forward the incident light with the angular spectrum method. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        '''
        field_input = light.field

        pad_width = compute_pad_width(field_input, linear)
        # Adjust frequency calculations based on 'linear'
        if linear:
            # 2x zero padding for the transfer function
            fx = np.arange(-light.dim[3], light.dim[3]) / (2*light.pitch * light.dim[3])
            fy = np.arange(-light.dim[2], light.dim[2]) / (2*light.pitch * light.dim[2])
        else:
            fx = np.arange(-light.dim[3]//2, light.dim[3]//2) / (light.pitch * light.dim[3])
            fy = np.arange(-light.dim[2]//2, light.dim[2]//2) / (light.pitch * light.dim[2])
        fxx, fyy = np.meshgrid(fx, fy, indexing='xy')

        k = 2*np.pi / light.wvl
        gamma = np.sqrt(np.abs(1. - (light.wvl*fxx)**2 - (light.wvl*fyy)**2))
        H = torch.from_numpy((np.exp(1j*k*float(z)*gamma)).astype(np.complex64)).to(light.device)
        H = torch.complex(H.real, H.imag)
        F = fft(field_input, pad_width=pad_width)
        G = F * H
        field_propagated = ifft(G, pad_width=pad_width)

        light_propagated = light.clone()
        light_propagated.set_field(field_propagated)
        return light_propagated

    def forward_BL_ASM(self, light, z, linear):
        '''
        Forward the incident light with the bandlimited angular spectrum method. 

        Args:
            light: incident light 
        Forward the incident light with the bandlimited angular spectrum method,
        supporting multi-wavelength channels.

        Args:
            light: incident light 
            z: propagation distance in meter. 
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        '''
        field_input = light.field

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        light_propagated = light.clone()  # Assuming this method properly duplicates the Light object

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])

            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')
            wvl = wavelengths[chan]
            k = 2*torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))

            Ideal_angle_limit = light.get_ideal_angle_limit()
            theta_limit = np.arctan(light.dim[3]*light.pitch/float(np.abs(z))) * 180 / np.pi

            if theta_limit < Ideal_angle_limit:
                #print('BL_ASM: theta_limit is smaller than Ideal_angle_limit. Activating the frequency filter!')
                uv_filter = (torch.abs(fxx) < np.sin(np.deg2rad(theta_limit)) / wvl) & \
                            (torch.abs(fyy) < np.sin(np.deg2rad(theta_limit)) / wvl)
                H = (torch.exp(1j*k*z*gamma) * uv_filter).to(light.device)
            else:
                H = torch.exp(1j*k*z*gamma).to(light.device)

            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H
            field_propagated = ifft(G, pad_width=pad_width)

            # Set the propagated field for the current channel
            light_propagated.set_field(field_propagated)

        return light_propagated


    def forward_RayleighSommerfeld(self, light, z, linear=True, target_plane=None):
            # RayleighSommerfeld is not FFT-based calculation, so window size of target_plane is not fixed. 
            
            field_input = light.field
            light_propagated = light.clone()
            
            k = 2 * np.pi / light.wvl
            R, C = light.dim[2], light.dim[3]
            x = torch.linspace(-C//2, C//2, C, dtype=torch.float64) * light.pitch
            y = torch.linspace(-R//2, R//2, R, dtype=torch.float64) * light.pitch
            xx, yy = torch.meshgrid(x, y)
            xx = xx.to(light.device)
            yy = yy.to(light.device)
            
            if target_plane:
                xx_t, yy_t, zz_t = target_plane 
                R, C = xx_t.shape[0], xx_t.shape[1]
            else: 
                xx_t, yy_t, zz_t = xx.clone(), yy.clone(), torch.full(xx.shape, z, dtype=torch.float64, device=light.device) 
            
            
            field_output = torch.zeros_like(field_input, dtype=torch.complex128)
    
            # Iterate over each pixel in the output image
            for idx_x in range(C):
                for idx_y in range(R):
                    X, Y, Z = xx_t[idx_y, idx_x], yy_t[idx_y, idx_x], zz_t[idx_y, idx_x]
                    r = torch.sqrt((X-xx)**2 + (Y-yy)**2 + Z**2)
    
                    contribution = (Z/r) * (1-1./(1j*k*r)) * torch.exp(1j*k*r) / r * field_input 
                    field_output[:, :, idx_y, idx_x] = torch.sum(contribution)
                    
            
            field_output = k / (2*torch.pi*1j) * light.pitch**2 * field_output        
    
            light_propagated.field = field_output
            return light_propagated