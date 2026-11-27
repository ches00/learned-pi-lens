import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .math import wrap_phase


class OpticalElement:
    def __init__(self, dim, pitch, wvl, field_change=None, device='cpu', name="not defined", polar='non'):
        """
        Base class for optical elements. Any optical element change the wavefront of incident light
        The change of the wavefront is stored as amplitude and phase tensors
        Note that he number of channels is one for the wavefront modulation.
        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            field_change: [batch_size, # of channels, row, column] tensor of wavefront change, default is None
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
            name: name of the current optical element
        """

        self.name = name
        self.dim = dim
        self.pitch = pitch
        self.device = device
        if field_change is None:
            self.field_change = torch.ones(dim, dtype=torch.cfloat, device=device)
        else:
            self.field_change = field_change
        self.wvl = wvl
        self.polar = polar

    def shape(self):
        """
        Returns the shape of light-wavefront modulation. The nunmber of channels is one
        Returns:
            shape
        """
        return self.dim     

    def set_pitch(self, pitch):
        """
        Set the pixel pitch of the complex tensor
        Args:
            pitch: pixel pitch in meter
        """
        self.pitch = pitch

    def resize(self, target_pitch, interp_mode='nearest'):
        '''
        Resize the wavefront change by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''

        scale_factor = self.pitch / target_pitch
        amplitude_change = F.interpolate(self.get_amplitude_change(), scale_factor=scale_factor, mode=interp_mode)
        phase_change = F.interpolate(self.get_phase_change(), scale_factor=scale_factor, mode=interp_mode)
        self.field_change = amplitude_change * torch.exp(phase_change * 1j)
        self.dim = torch.Size((self.dim[0], self.dim[1], self.field_change.shape[2], self.field_change.shape[3]))
        self.set_pitch(target_pitch)

    def get_amplitude_change(self):
        '''
        Return the amplitude change of the wavefront
        Returns:
            amplitude change: ampiltude change
        '''

        return self.field_change.abs()

    def get_phase_change(self):
        '''
        Return the phase change of the wavefront
        Returns:
            phase change: phase change
        '''
        
        return self.field_change.angle()

    def set_field_change(self, field_change):
        """
        Set the field change 
        Args:
            field change: field change in a complex tensor
        """

        self.field_change = field_change


    def set_amplitude_change(self, amplitude):
        """
        Set the amplitude change 
        Args:
            amplitude change: amplitude change in the polar representation of the complex number 
        """

        phase = self.field_change.angle()
        self.field_change = amplitude*torch.exp(phase*1j)


    def set_phase_change(self, phase):
        """
        Set the phase change 
        Args:
            phase change: phase change in the polar representation of the complex number 
        """

        amplitude = self.field_change.abs()
        self.field_change = amplitude*torch.exp(phase*1j)

    def set_polar(self, polar):
        """
        Set the polarization boolean
        Args:
            polar: decide whether input light is polarized or not. 'non' or 'polar'
        """

        self.polar = polar

    def pad(self, pad_width, padval=0):
        """
        Pad the wavefront change with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """
        if padval == 0:
            self.field_change = torch.nn.functional.pad(self.field_change, pad_width)
        else:
            return NotImplementedError('only zero padding supported')

        self.dim = list(self.dim)
        self.dim[2], self.dim[3] = self.dim[2]+pad_width[0]+pad_width[1], self.dim[3]+pad_width[2]+pad_width[3]
        self.dim = tuple(self.dim)

    def forward(self, light, interp_mode='nearest'):
        """
        Forward the incident light with the optical element. 
        Args:
            light: incident light 
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        Returns:
            light after interaction with the optical element
        """
        if light.pitch > self.pitch:
            light.resize(self.pitch)
            light.set_pitch(self.pitch)
        elif light.pitch < self.pitch:
            self.resize(light.pitch)
            self.set_pitch(light.pitch)

        if self.polar=='non':
            return self.forwardNonPolar(light, interp_mode)
        elif self.polar=='polar':
            x = self.forwardNonPolar(light.get_lightX(), interp_mode)
            y = self.forwardNonPolar(light.get_lightY(), interp_mode)
            light.set_lightX(x)
            light.set_lightY(y)
            return light
        else:
            NotImplementedError('Polar is not set.')

    def forwardNonPolar(self, light, interp_mode='nearest'):
        """
        Forward the incident light with the optical element. 
        Args:
            light: incident light 
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        Returns:
            light after interaction with the optical element
        """

        if light.pitch > self.pitch:
            light.resize(self.pitch, interp_mode)
            light.set_pitch(self.pitch)
        elif light.pitch < self.pitch:
            self.resize(light.pitch, interp_mode)
            self.set_pitch(light.pitch)

        if light.wvl != self.wvl:
            return NotImplementedError('wavelength should be same for light and optical elements')

        # make sure that light and optical element have the same resolution, i.e. pixel count, by padding the smaller one
        r1 = np.abs((light.dim[2] - self.dim[2])//2)
        r2 = np.abs(light.dim[2] - self.dim[2]) - r1
        pad_width = (r1, r2, 0, 0)
        if light.dim[2] > self.dim[2]:
            self.pad(pad_width)
        elif light.dim[2] < self.dim[2]:
            light.pad(pad_width)
        #print(light.field.shape)

        c1 = np.abs((light.dim[3] - self.dim[3])//2)
        c2 = np.abs(light.dim[3] - self.dim[3]) - c1
        pad_width = (0, 0, c1, c2)
        if light.dim[3] > self.dim[3]:
            self.pad(pad_width)
        elif light.dim[3] < self.dim[3]:
            light.pad(pad_width)

        light.set_field(light.field*self.field_change)

        return light

    def visualize(self, b=0):
        """
        Visualize the wavefront modulation of the optical element
        Args:
            b: batch index to visualize default is 0
        """

        plt.figure(figsize=(13,6))
        plt.subplot(121)
        
        plt.imshow(self.get_amplitude_change().data.cpu()[b,...].squeeze())
        plt.title('amplitude')
        plt.colorbar()
        
        plt.subplot(122)
        plt.imshow(self.get_phase_change().data.cpu()[b,...].squeeze())
        plt.title('phase')
        plt.colorbar()
        
        plt.suptitle('%s, (%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s'
                    %(self.name, self.dim[2], self.dim[3], self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.show()


class RefractiveLens(OpticalElement):
    def __init__(self, dim, pitch, focal_length, wvl, device, polar='non'):
        """
        Thin refractive lens
        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            focal_length: focal length of the lens in meter
            wvl: wavelength of light in meter
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """

        super().__init__(dim, pitch, wvl, None, device, name="refractive_lens", polar=polar)

        self.focal_length = None

        self.set_focal_length(focal_length)
        self.set_phase_change( self.compute_phase(self.wvl, shift_x=0, shift_y=0) )

    def set_focal_length(self, focal_length):
        """
        Set the focal length of the lens
        Args:
            focal_length: focal length in meter 
        """

        self.focal_length = focal_length

    def compute_phase(self, wvl, shift_x=0, shift_y=0):
        """
        Set the phase of a thin lens
        Args:
            wvl: wavelength of light in meter
            shift_x: x displacement of the lens w.r.t. incident light
            shift_y: y displacement of the lens w.r.t. incident light
        """

        bw_R = self.dim[2]*self.pitch
        bw_C = self.dim[3]*self.pitch

        x = np.arange(-self.dim[3]/2, self.dim[3]/2) * self.pitch
        y = np.arange(-self.dim[2]/2, self.dim[2]/2) * self.pitch
        xx, yy = np.meshgrid(x, y, indexing='xy')

        theta_change = torch.tensor((-2*np.pi / wvl)*((xx-shift_x)**2 + (yy-shift_y)**2), device=self.device) / (2*self.focal_length)
        theta_change = (theta_change + np.pi) % (np.pi * 2) - np.pi
        theta_change = torch.unsqueeze(torch.unsqueeze(theta_change, axis=0), axis=0)
        
        return theta_change


def height2phase(height, wvl, RI, wrap=True):
    """
    Convert the height of a material to the corresponding phase shift 
    Args:
        height: height of the material in meter
        wvl: wavelength of light in meter
        RI: refractive index of the material at the wavelength
        wrap: return the wrapped phase [0,2pi]
    Returns:
        phase: phase change induced by the height
    """
    dRI = RI - 1
    wv_n = 2. * np.pi / wvl
    phi = wv_n * dRI * height
    if wrap:
        phi = wrap_phase(phi, stay_positive=True)
    return phi

def phase2height(phase_u, wvl, RI, minh=None):
    """
    Convert the phase change to the height of a material.
    Note that the mapping from phase to height is not one to one.
    There is an integer-wrapping scalar:
        height = wvl/(RI-1) * (phase_u + i*2pi), where i is an integer
    So, this function takes the minimum height minh that can constrain the conversion
    Then, minimal height is chosen so that height is always greater than or equal to minh

    Args:
        phase_u: phase change of light 
        wvl: wavelength of light in meter
        RI: refractive index of the material at the wavelength
        minh: 
    Returns:
        height: height of the material that induces the phase change
    """
    dRI = RI - 1
    if minh is not None:
        i = torch.ceil(((dRI/wvl)*minh - phase_u)/(2*np.pi))
    else:
        i = 0
    height = wvl * (phase_u + 2*np.pi*i) / dRI
    return height


class DOE(OpticalElement):
    def __init__(self, dim, pitch, material, wvl, device, amplitude=None, height=None, field_change=None, polar='non'):
        """
        Diffractive optical element (DOE)
        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            material: material of the DOE
            wvl: wavelength of light in meter
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
            height: height map of the material in meter
            field_change: wavefront change of light 
        """

        super().__init__(dim, pitch, wvl, field_change, device, name="doe", polar=polar)
        self.material = material
        self.height = None

        if (height is None) and (field_change is not None):
            self.mode = 'field'
            phase = field_change.ang()
            self.set_field_change(field_change)
            if material != 'SiO2':
                height = phase2height(phase, wvl, material.get_RI(wvl))
                self.set_height(height)
        elif (height is not None) and (field_change is None):
            self.mode = 'height'
            self.set_height(height)
            if material != 'SiO2':
                phase = height2phase(height, wvl, material.get_RI(wvl))
                self.set_field_change(torch.exp(phase*1j))
                
        elif (height is None) and field_change is None:
            self.mode = 'phase'
            phase = torch.zeros(dim, device=self.device)
            self.set_phase_change(phase, wvl)
            self.set_field_change(torch.exp(phase*1j))
            
        # read duty-cycle mapping coefficients 
        if self.material == 'SiO2':
            file_path = './fitting.txt'
            f = open(file_path, 'r')
            
            data = f.read().split('\n')
            
            self.num_poly = int(data[1])
            ph2duty = data[2].split(' ')
            duty2ph = data[4].split(' ')
            self.ph2duty = torch.Tensor([float(i) for i in ph2duty]).to(device)
            self.duty2ph = torch.Tensor([float(i) for i in duty2ph]).to(device)

    # Implemented by Eunsue 
    def phase2duty(self, phase):
        phase_norm = (phase % (torch.pi * 2))/(torch.pi * 2) # 0 ~ 1 [*2pi]
        phase_norm = phase_norm -1*(phase_norm > 0.8) #78)
            
        duty = 0
        for i in range(self.num_poly+1):
            duty += phase_norm**(self.num_poly-i) * self.ph2duty[i]
        
        return duty # 0 ~ 1     
        
    
    def duty2phase(self, duty, wvl):
        wvl_norm = (wvl*1e+9 -800)/100. 
        
        p = self.duty2ph 
        phase_norm = p[0] + p[1]*wvl_norm + p[2]*duty + p[3]*wvl_norm**2 + p[4]*wvl_norm*duty + p[5]*duty**2 + p[6]*wvl_norm**3 + p[7]*wvl_norm**2 *duty + p[8]*wvl_norm*duty**2
        + p[9]*duty**3 + p[10]*wvl_norm**4 + p[11]*wvl_norm**3*duty + p[12]*wvl_norm**2*duty**2 + p[13]*wvl_norm*1*duty**3 + p[14]*duty**4 
        
        phase_norm = phase_norm + 1*(phase_norm < 0)
        phase = phase_norm * 2 * torch.pi 
        
        return phase 
    
    def change_wvl(self, wvl):
        """
        Change the wavelength of phase change
        Args:
            wvl: wavelength of phase change
        """
        
        if self.material == 'SiO2':
            duty = self.phase2duty(self.get_phase_change()) # phase 2 duty in 850nm
            self.wvl = wvl 
            phase = self.duty2phase(duty, wvl)
            self.set_phase_change(phase, self.wvl)
            # TODO : change transmittance value (== Amplitude) 
        else:
            height = self.get_height()
            self.wvl = wvl
            phase = height2phase(height, self.wvl, self.material.get_RI(self.wvl))
            self.set_phase_change(phase, self.wvl)

    def set_diffraction_grating_1d(self, slit_width, minh, maxh):
        """
        Set the wavefront modulation as 1D diffraction grating 
        Args:
            slit_width: width of slit in meter
            minh: minimum height in meter
            maxh: maximum height in meter
        """

        slit_width_px = np.round(slit_width / self.pitch)
        slit_space_px = slit_width_px

        dg = np.zeros((self.dim[2], self.dim[3]))
        slit_num_r = self.dim[2] // (2 * slit_width_px)
        slit_num_c = self.dim[3] // (2 * slit_width_px)

        dg[:] = minh

        for i in range(int(slit_num_c)):
            minc = int((slit_width_px + slit_space_px) * i)
            maxc = int(minc + slit_width_px)

            dg[:, minc:maxc] = maxh
        pc = torch.tensor(dg.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
        self.set_field_change(torch.exp(1j*pc))

    def set_diffraction_grating_2d(self, slit_width, minh, maxh):
        """
        Set the wavefront modulation as 2D diffraction grating 
        Args:
            slit_width: width of slit in meter
            minh: minimum height in meter
            maxh: maximum height in meter
        """

        slit_width_px = np.round(slit_width / self.pitch)
        slit_space_px = slit_width_px

        dg = np.zeros((self.dim[2], self.dim[3]))
        slit_num_r = self.dim[2] // (2 * slit_width_px)
        slit_num_c = self.dim[3] // (2 * slit_width_px)

        dg[:] = minh

        for i in range(int(slit_num_r)):
            for j in range(int(slit_num_c)):
                minc = int((slit_width_px + slit_space_px) * j)
                maxc = int(minc + slit_width_px)
                minr = int((slit_width_px + slit_space_px) * i)
                maxr = int(minr + slit_width_px)

                dg[minr:maxr, minc:maxc] = maxh

        pc = torch.tensor(dg.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
        self.set_field_change(torch.exp(1j*pc))

    def set_Fresnel_lens(self, focal_length, shift_x=0, shift_y=0):
        """
        Set the wavefront modulation as a fresnel lens 
        Args:
            focal_length: focal length in meter 
            shift_x: x displacement of the lens w.r.t. incident light
            shift_y: y displacement of the lens w.r.t. incident light
        """

        x = np.arange(-self.dim[3]*self.pitch/2, self.dim[3]*self.pitch/2, self.pitch)
        x = x[:self.dim[3]]
        y = np.arange(-self.dim[2]*self.pitch/2, self.dim[2]*self.pitch/2, self.pitch)
        y = y[:self.dim[2]]
        xx,yy = np.meshgrid(x,y, indexing='xy')
        xx = torch.tensor(xx, device=self.device)
        yy = torch.tensor(yy, device=self.device)

        phase_u = (-2*np.pi / self.wvl) * (torch.sqrt((xx-shift_x)**2 + (yy-shift_y)**2 + focal_length**2) - focal_length) #- focal_length)
        phase_w = wrap_phase(phase_u)
        phase_w = phase_w.unsqueeze(0).unsqueeze(0)

        self.set_field_change(torch.exp(1j*phase_w))
        #print('fresnel lens : ', self.field_change.device)

    def resize(self, target_pitch):
        '''
        Resize the wavefront by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
        '''
        scale_factor = self.pitch / target_pitch
        #super().resize(target_pitch)

        if self.mode == 'phase':
            super().resize(target_pitch)
        elif self.mode == 'height':
            self.set_height(F.interpolate(self.height, scale_factor=scale_factor, mode='bilinear', align_corners=False))
        else:
            NotImplementedError('Mode is not set.')

    def get_height(self):
        """
        Return the height map of the DOE
        Returns:
            height map: height map in meter
        """

        if self.mode == 'height':
            return self.height
        elif self.mode == 'phase':
            height = phase2height(self.get_phase_change(), self.wvl, self.material.get_RI(self.wvl))
            return height
        else:
            NotImplementedError('Mode is not set.')

    def get_phase_change(self):
        """
        Return the phase change induced by the DOE
        Returns:
            phase change: phase change
        """
        if self.mode == 'height':
            self.to_phase_mode()
        return self.field_change.angle()

    def set_field_change(self, field_change, sync_height=False):
        """
        Change the field change of the DOE
        Args:
            field change: field change in a complex tensor
            sync_height: sync the height of the DOE according to the phase change
        """
        super().set_field_change(field_change)

        if sync_height:
            height = phase2height(self.get_phase_change(), self.wvl, self.material.get_RI(self.wvl))
            self.set_height( height, sync_phase=False)

    def set_height(self, height):
        """
        Set the height map of the DOE
        Args:
            height map: height map in meter
        """

        if self.mode == 'height':
            self.height = height
        elif self.mode == 'phase':
            self.set_phase_change(height2phase(height, self.wvl, self.material.get_RI(self.wvl)), self.wvl)

    def set_phase_change(self, phase_change, wvl):
        """
        Set the phase change induced by the DOE
        Args:
            phase change: phase change
        """
        #print(self.mode)
        if self.mode == 'height':
            self.set_height(phase2height(phase_change, wvl, self.material.get_RI(wvl)))
        if self.mode == 'phase':
            self.wvl = wvl
            self.phase_change = phase_change
            amplitude = self.get_amplitude_change() 
            
            #self.set_field_change(torch.exp(1j*phase_w))
            self.set_field_change(amplitude * torch.exp(1j*phase_change))

    def to_phase_mode(self):
        """
        Change the mode to phase change
        """
        if self.mode == 'height':
            self.phase_change = height2phase(self.height, self.wvl, self.material.get_RI(self.wvl))
            self.mode = 'phase'
            self.height = None

    def to_height_mode(self):
        """
        Change the mode to height 
        """
        if self.mode == 'phase':
            self.height = phase2height(self.phase_change, self.wvl, self.material.get_RI(self.wvl))
            self.mode = 'height'


class SLM(OpticalElement):
    def __init__(self, dim, pitch, wvl, device, polar='non'):
        """
        Spatial light modulator (SLM)
        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """

        super().__init__(dim, pitch, wvl, device=device, name="SLM", polar=polar)

    def set_lens(self, focal_length, shift_x=0, shift_y=0):
        """
        Set the phase of a thin lens
        Args:
            wvl: wavelength of light in meter
            shift_x: x displacement of the lens w.r.t. incident light
            shift_y: y displacement of the lens w.r.t. incident light
        """

        x = np.arange(-self.dim[3]*self.pitch/2, self.dim[3]*self.pitch/2, self.pitch)
        y = np.arange(-self.dim[2]*self.pitch/2, self.dim[2]*self.pitch/2, self.pitch)
        xx,yy = np.meshgrid(x,y)

        phase_u = (2*np.pi / self.wvl)*((xx-shift_x)**2 + (yy-shift_y)**2) / (2*focal_length)
        phase_u = torch.tensor(phase_u.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
        phase_w = wrap_phase(phase_u, stay_positive=False)
        self.set_phase_change(phase_w)

    def set_amplitude_change(self, amplitude, wvl):
        """
        Set the amplitude change 
        Args:
            amplitude change: amplitude change in the polar representation of the complex number 
            wvl: wavelength of light in meter

        """
        self.wvl = wvl
        super().set_amplitude_change(amplitude)

    def set_phase_change(self, phase_change, wvl):
        """
        Set the phase change 
        Args:
            phase change: phase change in the polar representation of the complex number 
            wvl: wavelength of light in meter
        """
        self.wvl = wvl
        super().set_phase_change(phase_change)
        
        
class SLM2(OpticalElement):
    def __init__(self, dim, pitch, wvl, device):
        """
        SLM which can control phase & amplitude of each polarization component respectively
        Args:
            R: row
            C: column
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
            B: batch size
        """
        super().__init__(dim, pitch, wvl, device=device, name="Metasurface", polar='polar')
        self.amplitude_change = torch.ones((dim[0], 1, dim[2], dim[3], 2), device=self.device)
        self.phase_change = torch.zeros((dim[0], 1, dim[2], dim[3], 2), device=self.device)

    def set_amplitude_change(self, amplitude, wvl):
        """
        Set the amplitude change
        Args:
            amplitude change: [B, 1, R, C, 2] amplitude change in the polar representation of the complex number
        """
        self.wvl = wvl
        super().set_amplitude_change(amplitude)

    def set_phase_change(self, phase_change, wvl):
        """
        Set the phase change
        Args:
            phase change: [B, 1, R, C, 2] phase change in the polar representation of the complex number
        """
        self.wvl = wvl
        super().set_phase_change(phase_change)

    def set_amplitudeX_change(self, amplitude, wvl):
        """
        Set the amplitude change
        Args:
            amplitude change: [B, 1, R, C] amplitude change of the X component in the polar representation of the complex number
        """
        self.wvl = wvl
        amp = self.get_amplitude_change()
        amp[:,:,:,:,0] = amplitude
        super().set_amplitude_change(amp)

    def set_amplitudeY_change(self, amplitude, wvl):
        """
        Set the amplitude change
        Args:
            amplitude change: [B, 1, R, C] amplitude change of the Y component in the polar representation of the complex number
        """
        self.wvl = wvl
        amp = self.get_amplitude_change()
        amp[:,:,:,:,1] = amplitude
        super().set_amplitude_change(amp)

    def set_phaseX_change(self, phase_change, wvl):
        """
        Set the amplitude change
        Args:
            phase change: [B, 1, R, C] phase change of the X component in the polar representation of the complex number
        """
        self.wvl = wvl
        phase = self.get_phase_change()
        phase[:,:,:,:,0] = phase_change
        super().set_phase_change(phase)

    def set_phaseY_change(self, phase_change, wvl):
        """
        Set the amplitude change
        Args:
            phase change: [B, 1, R, C] phase change of the Y component in the polar representation of the complex number
        """
        self.wvl = wvl
        phase = self.get_phase_change()
        phase[:,:,:,:,1] = phase_change
        super().set_phase_change(phase)

    def get_phase_changeX(self):
        """
        Get the phase change of X
        Returns:
            phase change X: [B, 1, R, C] phase change of the X component
        """
        return self.get_phase_change()[:,:,:,:,0]

    def get_phase_changeY(self):
        """
        Get the phase change of Y
        Returns:
            phase change X: [B, 1, R, C] phase change of the Y component
        """
        return self.get_phase_change()[:,:,:,:,1]

    def get_amplitude_changeX(self):
        """
        Get the amplitude change of X
        Returns:
            amplitude change X: [B, 1, R, C] amplitude change of the X component
        """
        return self.get_amplitude_change()[:,:,:,:,0]

    def get_amplitude_changeY(self):
        """
        Get the amplitude change of Y
        Returns:
            amplitude change X: [B, 1, R, C] amplitude change of the Y component
        """
        return self.get_amplitude_change()[:,:,:,:,1]
        
    def forward(self, light, interp_mode='nearest'):
        if light.wvl != self.wvl:
            return NotImplementedError('wavelength should be same for light and optical elements')
        
        if light.pitch > self.pitch:
            light.resize(self.pitch)
            light.set_pitch(self.pitch)
        elif light.pitch < self.pitch:
            self.resize(light.pitch)
            self.set_pitch(light.pitch)
            
        r1 = np.abs((light.dim[2] - self.dim[2])//2)
        r2 = np.abs(light.dim[2] - self.dim[2]) - r1
        pad_width = (r1, r2, 0, 0)
        if light.dim[2] > self.dim[2]:
            self.pad(pad_width)
        elif light.dim[2] < self.dim[2]:
            light.pad(pad_width)

        c1 = np.abs((light.dim[3] - self.dim[3])//2)
        c2 = np.abs(light.dim[3] - self.dim[3]) - c1
        pad_width = (0, 0, c1, c2)
        if light.dim[3] > self.dim[3]:
            self.pad(pad_width)
        elif light.dim[3] < self.dim[3]:
            light.pad(pad_width)
        
        phase = (light.get_phase()[..., 0] + self.get_phase_change() + np.pi) %(np.pi * 2) - np.pi  #(light.get_phase()[...,0] + self.get_phase_change() +np.pi) % (np.pi*2) - np.pi
        light.set_phaseX(phase[...,0])
        light.set_phaseY(phase[...,1])
        light.set_amplitudeX(light.get_amplitudeX() * self.get_amplitude_change()[...,0])
        light.set_amplitudeY(light.get_amplitudeY() * self.get_amplitude_change()[...,1])
        #print(light.device)
        return light

    def pad(self, pad_width, padval=0):
        """
        Pad the wavefront change with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """
        if padval == 0:
            self.amplitude_change = torch.nn.functional.pad(self.get_amplitude_change(), (0,0,0,0,pad_width[2],pad_width[3],pad_width[0],pad_width[1]), value=1)
            self.phase_change = torch.nn.functional.pad(self.get_phase_change(), (0,0,0,0,pad_width[2],pad_width[3],pad_width[0],pad_width[1]))
        else:
            return NotImplementedError('only zero padding supported')

        self.R += pad_width[0] + pad_width[1]
        self.C += pad_width[2] + pad_width[3]
        
    def visualize(self, b=0):
        """
        Visualize the wavefront modulation of the optical element
        Args:
            b: batch index to visualize default is 0
        """

        plt.figure(figsize=(13,8))
        
        plt.subplot(221)
        plt.imshow(self.get_amplitude_changeX().data.cpu()[b,...].squeeze())
        plt.title('amplitude X')
        plt.colorbar()
        
        plt.subplot(222)
        plt.imshow(self.get_phase_changeX().data.cpu()[b,...].squeeze())
        plt.title('phase X')
        plt.colorbar()
        
        plt.subplot(223)
        plt.imshow(self.get_amplitude_changeY().data.cpu()[b,...].squeeze())
        plt.title('amplitude Y')
        plt.colorbar()
        
        plt.subplot(224)
        plt.imshow(self.get_phase_changeY().data.cpu()[b,...].squeeze())
        plt.title('phase Y')
        plt.colorbar()
        
        plt.suptitle('%s, (%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s'
                    %(self.name, self.dim[2], self.dim[3], self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.show()


class Aperture(OpticalElement):
    def __init__(self, dim, pitch, aperture_diameter, aperture_shape, wvl, device='cpu', polar='non'):
        """
        Aperture
        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            aperture_diameter: diamater of the aperture in meter
            aperture_shape: shape of the aperture. {'square', 'circle'}
            wvl: wavelength of light in meter
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """

        super().__init__(dim, pitch, wvl, device=device, name="aperture", polar=polar)

        self.aperture_diameter = aperture_diameter
        self.aperture_shape = aperture_shape
        self.amplitude_change = torch.zeros((self.dim[2], self.dim[3]), device=device)
        if self.aperture_shape == 'square':
            self.set_square()
        elif self.aperture_shape == 'circle':
            self.set_circle()
        else:
            return NotImplementedError

    def set_square(self):
        """
        Set the amplitude modulation of the aperture as square
        """

        self.aperture_shape = 'square'

        [x, y] = np.mgrid[-self.dim[2]//2:self.dim[2]//2, -self.dim[3]//2:self.dim[3]//2].astype(np.float32)
        r = self.pitch * np.asarray([abs(x), abs(y)]).max(axis=0)
        r = np.expand_dims(np.expand_dims(r, axis=0), axis=0)

        max_val = self.aperture_diameter / 2
        amp = (r <= max_val).astype(np.float32)
        amp[amp == 0] = 1e-20  # to enable stable learning
        self.set_field_change(torch.tensor(amp, device=self.device))
    def set_circle(self, cx=0, cy=0, dia=None):
        """
        Set the amplitude modulation of the aperture as circle
        Args:
            cx, cy: relative center position of the circle with respect to the center of the light wavefront
            dia: circle diameter
        """
        [x, y] = np.mgrid[-self.dim[2]//2:self.dim[2]//2, -self.dim[3]//2:self.dim[3]//2].astype(np.float32)
        r2 = (x-cx) ** 2 + (y-cy) ** 2
        r2[r2 < 0] = 1e-20
        r = self.pitch * np.sqrt(r2)
        r = np.expand_dims(np.expand_dims(r, axis=0), axis=0)
        
        if dia is not None:
            self.aperture_diameter = dia
        self.aperture_shape = 'circle'
        max_val = self.aperture_diameter / 2
        amp = (r <= max_val).astype(np.float32)
        amp[amp == 0] = 1e-20
        self.set_field_change(torch.tensor(amp, device=self.device))


def quantize(x, levels, vmin=None, vmax=None, include_vmax=True):
    """
    Quantize the floating array
    Args:
        levels: number of quantization levels 
        vmin: minimum value for quantization
        vmax: maximum value for quantization
        include_vmax: include vmax for the quantized levels
            False: quantize x with the space of 1/levels-1.  
            True: quantize x with the space of 1/levels
    """

    if include_vmax is False:
        if levels == 0:
            return x

        if vmin is None:
            vmin = x.min()
        if vmax is None:
            vmax = x.max()

        #assert(vmin <= vmax)

        normalized = (x - vmin) / (vmax - vmin + 1e-16)
        if type(x) is np.ndarray:
            levelized = np.floor(normalized * levels) / (levels - 1)
        elif type(x) is torch.tensor:    
            levelized = (normalized * levels).floor() / (levels - 1)
        result = levelized * (vmax - vmin) + vmin
        result[result < vmin] = vmin
        result[result > vmax] = vmax
    
    elif include_vmax is True:
        space = (x.max()-x.min())/levels
        vmin = x.min()
        vmax = vmin + space*(levels-1)
        if type(x) is np.ndarray:
            result = (np.floor((x-vmin)/space))*space + vmin
        elif type(x) is torch.tensor:    
            result = (((x-vmin)/space).floor())*space + vmin
        result[result<vmin] = vmin
        result[result>vmax] = vmax
    
    return result