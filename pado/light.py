import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

class Light:
    def __init__(self, dim, pitch, wvl, field=None, device='cpu'):
        """
        Light wave that has a complex field as a wavefront

        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            field: [batch_size, # of channels, row, column] tensor of wavefront, default is None
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """

        self.dim = dim
        self.pitch = pitch
        self.device = device
        self.wvl = wvl
        
        if field is None:
            field = torch.ones(dim, device=device, dtype=torch.cfloat)
        self.field = field
        
    def crop(self, crop_width):
        """
        Crop the light wavefront by crop_width
        Args:
            crop_width: (tuple) crop width of the tensor following torch functional pad 
        """

        self.field = self.field[...,crop_width[2]:-crop_width[3], crop_width[0]:-crop_width[1]]
        self.dim[2], self.dim[3] = self.field.shape[2], self.field.shape[3]

    def clone(self):
        """
        Clone the light and return it
        """

        return Light(self.dim, self.pitch, self.wvl, self.field.clone(), device=self.device)


    def pad(self, pad_width, padval=0):
        """
        Pad the light wavefront with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """

        if padval == 0:
            self.field = torch.nn.functional.pad(self.field, pad_width)
        else:
            return NotImplementedError('only zero padding supported')
        self.dim[2], self.dim[3] = self.dim[2]+pad_width[0]+pad_width[1], self.dim[3]+pad_width[2]+pad_width[3]

    def set_real(self, real):
        """
        Set the real part of the light wavefront
        Args:
            real: real part in the rect representation of the complex number 
        """

        self.field.real = real 

    def set_imag(self, imag):
        """
        Set the imaginary part of the light wavefront
        Args:
            imag: imaginary part in the rect representation of the complex number 
        """
        self.field.imag = imag
        
    def set_amplitude(self, amplitude):
        """
        Set the amplitude of the light wavefront
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        phase = self.field.angle()
        self.field = amplitude*torch.exp(phase*1j)

    def set_phase(self, phase):
        """
        Set the phase of the complex tensor
        Args:
            phase: phase in the polar representation of the complex number 
        """
        amplitude = self.field.abs()
        self.field = amplitude*torch.exp(phase*1j)


    def set_field(self, field):
        """
        Set the wavefront modulation of the complex tensor
        Args:
            field: wavefront as a complex number
        """
        self.field = field

    def set_pitch(self, pitch):
        """
        Set the pixel pitch of the complex tensor
        Args:
            pitch: pixel pitch in meter
        """
        self.pitch = pitch


    def get_amplitude(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """

        return self.field.abs()

    def get_phase(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """

        return self.field.angle()

    def get_field(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """

        return self.field

    def get_intensity(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return (self.field * torch.conj(self.field)).real

    def get_bandwidth(self):
        """
        Return the bandwidth of light wavefront
        Returns:
            R_m: spatial height of the wavefront 
            C_m: spatial width of the wavefront 
        """

        return self.pitch*self.dim[2], self.pitch*self.dim[3]
    
    def get_ideal_angle_limit(self):
        """
        Return the ideal angle limit of light wavefront based on the optical axis.
        Returns:
            Ideal_angle_limit: ideal angle limit of light wavefront (deg)
        """

        sin_val = np.minimum((self.wvl / self.pitch) * 0.5, 1.)
        Ideal_angle_limit = np.arcsin(sin_val) * 180 / np.pi
        
        return Ideal_angle_limit - 0.0001

    def magnify(self, scale_factor, interp_mode='nearest'):
        '''
        Change the wavefront resolution without changing the pixel pitch
        Args:
            scale_factor: scale factor for interpolation used in tensor.nn.functional.interpolate
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        self.field.real = F.interpolate(self.field.real, scale_factor=scale_factor, mode=interp_mode)
        self.field.imag = F.interpolate(self.field.imag, scale_factor=scale_factor, mode=interp_mode)
        self.dim[2], self.dim[3] = self.field.shape[2], self.field.shape[3]


    def resize(self, target_pitch, interp_mode='nearest'):
        '''
        Resize the wavefront by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        scale_factor = self.pitch / target_pitch
        self.magnify(scale_factor, interp_mode)
        self.set_pitch(target_pitch)

    def set_spherical_light(self, z, dx=0, dy=0, wvls=None):
        '''
        Set the wavefront as spherical one coming from the position of (dx,dy,z). 
        Args:
            z: z distance of the spherical light source from the current light position # (N, 1)
            dx: x distance of the spherical light source from the current light position # (N, 1)
            dy: y distance of the spherical light source from the current light position # (N, 1)
        '''
        batch = self.dim[0] # N  
        z = z[:, None, None, None]

        x = torch.arange(-self.dim[3] // 2, self.dim[3] // 2, dtype=torch.float64, device=self.device)
        y = torch.arange(-self.dim[2] // 2, self.dim[2] // 2, dtype=torch.float64, device=self.device)
        y, x = torch.meshgrid(y, x)
        #[x, y] = np.mgrid[-self.dim[3] // 2:self.dim[3] // 2, -self.dim[2] // 2:self.dim[2] // 2].astype(np.float64)
        
        x = x[None, None, :, :] #.unsqueeze(0).unsqueeze(0) # 1 x 1 x H x W 
        y = y[None, None, :, :] #.unsuqeeze(0).unsqueeze(0) # 1 x 1 x H x W 
        
        dx = dx[:,  None, None, None] # N x 1 x 1 x 1 
        dy = dy[:,  None, None, None]
        
        x = x * self.pitch
        y = y * self.pitch

        r = torch.sqrt((x-dx)**2 + (y-dy)**2 + z**2)
        if wvls is not None:
            theta = 2 * torch.pi * r / wvls 
        else:
            theta = 2 * torch.pi * r / self.wvl
        #r = np.sqrt((x - dx) ** 2 + (y - dy) ** 2 + z ** 2)  # this is computed in double precision
        #theta = 2 * np.pi * r / self.wvl
        theta = theta % (2*torch.pi)
        #theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)%(2*np.pi)
        theta = theta.to(device=self.device, dtype=torch.float32) #theta.astype(np.float32)

        #theta = torch.tensor(theta, device=self.device)
        mag = torch.ones_like(theta)

        self.set_field(mag*torch.exp(theta*1j))
        
    def set_xy_plane_light(self, theta, wvls=None):
        #theta_ = theta[:, None, None, None] # N x 1 x 1 x 1
        R, C = self.dim[2], self.dim[3]
        amplitude = torch.ones((1, 1, self.dim[2], self.dim[3]), device=self.device)
        phase = torch.zeros((1, 1, self.dim[2], self.dim[3]), device=self.device)
        
        dx = torch.linspace(-C * self.pitch / 2,C *self.pitch / 2 , R).to(self.device)
        dy = torch.linspace(-R * self.pitch / 2, R * self.pitch/ 2 , C).to(self.device)
        dx, dy = torch.meshgrid(dx, dy, indexing='xy') # R x C 
        
        dx = dx[None, None, :, :]
        dy = dy[None, None, :, :]
        
        theta_x, theta_y = theta # 
        print('hi')
        #r = torch.sqrt(dx**2 + dy**2)
        
        if wvls is not None:
            term = -2 * torch.pi * (dx * torch.sin(torch.deg2rad(theta_x))  + dy * torch.sin(torch.deg2rad(theta_y)))/ wvls  #theta = 2 * torch.pi * r / wvls 
        else:
            term = -2 * torch.pi * (dx * torch.sin(torch.deg2rad(theta_x)) + dy * torch.sin(torch.deg2rad(theta_y))) / self.wvl
            
        phase = phase - term #- 2 * torch.pi * dx * torch.sin(theta_) / self.wvl #r * torch.sin(theta_)
        
        self.set_field(amplitude*torch.exp(phase*1j))
        

    def set_plane_light(self, theta, wvls=None):
        '''
        Set the wavefront as a plane wave with zero phase and amptliude of one
        '''
        
        #theta_ = theta[:, None, None, None] # N x 1 x 1 x 1
        R, C = self.dim[2], self.dim[3]
        amplitude = torch.ones((1, 1, self.dim[2], self.dim[3]), device=self.device)
        phase = torch.zeros((1, 1, self.dim[2], self.dim[3]), device=self.device)
        
        dx = torch.linspace(-C * self.pitch / 2,C *self.pitch / 2 , R).to(self.device)
        dy = torch.linspace(-R * self.pitch / 2, R * self.pitch/ 2 , C).to(self.device)
        dx, dy = torch.meshgrid(dx, dy) # R x C 
        
        dx = dx[None, None, :, :]
        dy = dy[None, None, :, :]
        
        #r = torch.sqrt(dx**2 + dy**2)
        
        if wvls is not None:
            term = -2 * torch.pi * dx * torch.sin(torch.deg2rad(theta)) / wvls  #theta = 2 * torch.pi * r / wvls 
        else:
            term = -2 * torch.pi * dx * torch.sin(torch.deg2rad(theta)) / self.wvl
            
        phase = phase - term #- 2 * torch.pi * dx * torch.sin(theta_) / self.wvl #r * torch.sin(theta_)
        
        self.set_field(amplitude*torch.exp(phase*1j))

    def save(self, fn):
        '''
        Save the amplitude and phase of the light wavefront as a file
        Args:
            fn: filename to save. the format should be either "npy" or "mat"

        '''
        field_np = self.get_field().data.cpu().numpy()
        if fn[-3:] == 'npy':
            np.save(fn, field_np)
        elif fn[-3:] == 'mat':
            savemat(fn, {'field':field_np})
        else:
            print('extension in %s is unknown'%fn)
        print('light saved to %s\n'%fn)

    def visualize(self,b=0,c=0):
        """
        Visualize the light wave 
        Args:
            b: batch index to visualize default is 0
            c: channel index to visualize. default is 0
        """

        bw = self.get_bandwidth()
        std = 3
        
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        amplitude_b = self.get_amplitude().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude_b,
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno')
        # plt.imshow(amplitude_b,
        #         extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', 
        #         vmin=0, vmax=amplitude_b.mean() + amplitude_b.std() * std)
        plt.title('amplitude')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(132)
        phase = self.get_phase().data.cpu()[b,c,...].squeeze()
        plt.imshow(phase,
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(133)
        intensity_b = self.get_intensity().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity_b,
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno')
        # plt.imshow(intensity_b,
        #            extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', 
        #            vmin=0, vmax=intensity_b.mean() + intensity_b.std() * std)
        plt.title('intensity')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.suptitle('(%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s'%(self.dim[2], self.dim[3],
                                                                         self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.show()

    def shape(self):
        """
        Returns the shape of light wavefront
        Returns:
            shape
        """
        return self.field.shape()

class PolarizedLight(Light):
    def __init__(self, dim, pitch, wvl, fieldX=None, fieldY=None, device='cpu'):
        """
        Light wave that has a polarized complex field as a wavefront

        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            fieldX: [batch_size, # of channels, row, column] tensor of wavefront of X component, default is None
            fieldY: [batch_size, # of channels, row, column] tensor of wavefront of Y component, default is None
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """
        self.dim = dim
        self.pitch = pitch
        self.device = device
        self.wvl = wvl
        
        fieldX = torch.ones(dim, device=device, dtype=torch.cfloat) if fieldX is None else fieldX 
        fieldY = torch.ones(dim, device=device, dtype=torch.cfloat) if fieldY is None else fieldY 
        self.lightX = Light(dim, pitch, wvl, fieldX, device)
        self.lightY = Light(dim, pitch, wvl, fieldY, device)

    def get_fieldX(self):
        return self.lightX.get_field()

    def get_fieldY(self):
        return self.lightY.get_field()
     
    def crop(self, crop_width):
        """
        Crop the light wavefront by crop_width
        Args:
            crop_width: (tuple) crop width of the tensor following torch functional pad 
        """
        self.lightX.crop(crop_width)
        self.lightY.crop(crop_width)
        
        self.dim[2], self.dim[3] = self.lightX.dim[2], self.lightX.dim[3]
        
    def clone(self):
        """
        Clone the light and return it
        """
        return PolarizedLight(self.dim, self.pitch, self.wvl, self.get_fieldX().clone(), self.get_fieldX().clone(), device=self.device)
        
    def pad(self, pad_width, padval=0):
        """
        Pad the light wavefront with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """
        if padval == 0:
            self.lightX.pad(pad_width)
            self.lightY.pad(pad_width)
        else:
            return NotImplementedError('only zero padding supported')

        self.dim[2], self.dim[3] = self.dim[2]+pad_width[0]+pad_width[1], self.dim[3]+pad_width[2]+pad_width[3]

    def set_realX(self, real):
        """
        Set the real part of the light wavefront of X component
        Args:
            real: real part in the rect representation of the complex number 
        """
        self.lightX.set_real(real)
        
    def set_imagX(self, imag):
        """
        Set the imaginary part of the light wavefront of X component
        Args:
            imag: imaginary part in the rect klrepresentation of the complex number
        """
        self.lightX.set_imag(imag)
        
    def set_amplitudeX(self, amplitude):
        """
        Set the amplitude of the light wavefront of X component
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        self.lightX.set_amplitude(amplitude)

    def set_phaseX(self, phase):
        """
        Set the phase of the complex tensor of X component
        Args:
            phase: phase in the polar representation of the complex number 
        """
        self.lightX.set_phase(phase)

    def set_fieldX(self, field):
        """
        Set the wavefront modulation of the complex tensor of X component
        Args:
            field: wavefront as a complex number
        """
        self.lightX.set_field(field)

    def set_realY(self, real):
        """
        Set the real part of the light wavefront of Y component
        Args:
            real: real part in the rect representation of the complex number 
        """
        self.lightY.set_real(real)
        
    def set_imagY(self, imag):
        """
        Set the imaginary part of the light wavefront of Y component
        Args:
            imag: imaginary part in the rect klrepresentation of the complex number
        """
        self.lightY.set_imag(imag)
        
    def set_amplitudeY(self, amplitude):
        """
        Set the amplitude of the light wavefront of Y component
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        self.lightY.set_amplitude(amplitude)

    def set_phaseY(self, phase):
        """
        Set the phase of the complex tensor of Y component
        Args:
            phase: phase in the polar representation of the complex number 
        """
        self.lightY.set_phase(phase)

    def set_fieldY(self, field):
        """
        Set the wavefront modulation of the complex tensor of Y component
        Args:
            field: wavefront as a complex number
        """
        self.lightY.set_field(field)

    def set_pitch(self, pitch):
        """
        Set the pixel pitch of the complex tensor
        Args:
            pitch: pixel pitch in meter
        """
        self.pitch = pitch

    def get_amplitude(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        return self.get_intensity() ** 0.5
    
    def get_phase(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        x = self.lightX.get_phase()
        y = self.lightY.get_phase()
        return torch.stack((x, y), -1).unsqueeze(-1)
    
    def get_field(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        x = self.lightX.field()
        y = self.lightY.field()
        return torch.stack((x, y), -1).unsqueeze(-1)

    def get_intensity(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.lightX.get_intensity() + self.lightY.get_intensity()
        
    def get_bandwidth(self):
        """
        Return the bandwidth of light wavefront
        Returns:
            R_m: spatial height of the wavefront 
            C_m: spatial width of the wavefront 
        """
        return self.pitch*self.dim[2], self.pitch*self.dim[3]
        

    def get_amplitudeX(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        return self.lightX.get_amplitude()

    def get_phaseX(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        return self.lightX.get_phase()

    def get_fieldX(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        return self.lightX.get_field()

    def get_intensityX(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.lightX.get_intensity()
    
    def get_amplitudeY(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        return self.lightY.get_amplitude()

    def get_phaseY(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        return self.lightY.get_phase()

    def get_fieldY(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        return self.lightY.get_field()

    def get_intensityY(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.lightY.get_intensity()
    
    def get_lightX(self):
        """
        Return the light of X component
        Returns:
            light: light of X component
        """
        return self.lightX
    
    def get_lightY(self):
        """
        Return the light of Y component
        Returns:
            light: light of Y component
        """
        return self.lightY
    
    def set_lightX(self, light):
        """
        Set the light of X component
        Args:
            light: light of X component
        """
        self.lightX = light
    
    def set_lightY(self, light):
        """
        Set the light of Y component
        Args:
            light: light of Y component
        """
        self.lightY = light
    
    def set_field(self, field):
        """
        Set the field of light
        Args:
            field: field of light with 2 components
        """
        self.lightX.set_field(field[0])
        self.lightY.set_field(field[1])
    
    def magnify(self, scale_factor, interp_mode='nearest'):
        '''
        Change the wavefront resolution without changing the pixel pitch
        Args:
            scale_factor: scale factor for interpolation used in tensor.nn.functional.interpolate
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        self.lightX.magnify(scale_factor, interp_mode)
        self.lightY.magnify(scale_factor, interp_mode)
        self.dim[2], self.dim[3] = self.lightX.dim[2], self.lightY.dim[3]

    def resize(self, target_pitch, interp_mode='nearest'):
        '''
        Resize the wavefront by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        scale_factor = self.pitch / target_pitch
        self.magnify(scale_factor, interp_mode)
        self.set_pitch(target_pitch)

    def set_spherical_light(self, z, dx=0, dy=0):
        '''
        Set the wavefront as spherical one coming from the position of (dx,dy,z). 
        Args:
            z: z distance of the spherical light source from the current light position
            dx: x distance of the spherical light source from the current light position
            dy: y distance of the spherical light source from the current light position
        '''
        self.lightX.set_spherical_light(z, dx, dy)
        self.lightY.set_spherical_light(z, dx, dy)

    def set_plane_light(self):
        '''
        Set the wavefront as a plane wave with zero phase and amptliude of one
        '''
        self.lightX.set_plane_light()
        self.lightY.set_plane_light()
        
    def visualize(self,b=0,c=0):
        """
        Visualize the light wave 
        Args:
            b: batch index to visualize default is 0
            c: channel index to visualize. default is 0
        """
        bw = self.get_bandwidth()
        std = 3
        
        plt.figure(figsize=(15, 11))
        plt.subplot(331)
        amplitude_x = self.get_amplitudeX().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude_x, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', 
                   vmin=0, vmax=amplitude_x.mean() + amplitude_x.std() * std)
        plt.title('amplitude X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(332)
        phase_x = self.get_phaseX().data.cpu()[b,c,...].squeeze()
        plt.imshow(phase_x.squeeze(),
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(333)
        intensity_x = self.get_intensityX().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity_x, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_x.mean() + intensity_x.std() * std)
        plt.title('intensity X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.subplot(334)
        amplitude_y = self.get_amplitudeY().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude_y, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=amplitude_y.mean() + amplitude_y.std() * std)
        plt.title('amplitude Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(335)
        phase_y = self.get_phaseY().data.cpu()[b,c,...].squeeze()
        plt.imshow(phase_y.squeeze(),
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(336)
        intensity_y = self.get_intensityY().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity_y, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_y.mean() + intensity_y.std() * std)
        plt.title('intensity Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.subplot(337)
        amplitude = self.get_amplitude().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=amplitude.mean() + amplitude.std() * std)
        plt.title('amplitude')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(338)
        ratio = amplitude_x / amplitude_y
        plt.imshow(ratio.squeeze(),
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='gray',
                   vmin=ratio.mean()-ratio.std()*5, vmax=ratio.mean()+ratio.std()*5)  # cyclic colormap
        plt.title('Ratio (X / Y)')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(339)
        intensity = self.get_intensity().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity.mean() + intensity.std() * std)
        plt.title('intensity')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.suptitle('(%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s'%(self.dim[2], self.dim[3],
                                                                         self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.show()

    def shape(self):
        """
        Returns the shape of light wavefront
        Returns:
            shape
        """
        return self.lightX.get_field().shape()
