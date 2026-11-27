import torch
import numpy as np
import torch.nn.functional as F

"""
# At Pado, all measurements adhere to the International System of Units (SI).
"""
nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

s = 1
ms = 1e-3
us = ms * 1e-3
ns = us * 1e-3

def wrap_phase(phase_u, stay_positive=False):
    """
    Return the wrapped phase in the range [-pi, pi] or [0, 2pi].
    Args:
        phase_u: tensor of unwrapped phase
        stay_positive: keep the wrapped phase to be positive. Default is False
    Retruns:
        phase_w: wrapped phase
    """
    phase = phase_u%(2*np.pi)
    if stay_positive == False:
        phase[phase>torch.pi] -= (2*np.pi)
    return phase



def fft(arr_c, normalized="backward", pad_width=None, padval=0, shift=True):
    """
    Compute the Fast Fourier transform of a complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        normalized: Normalize the FFT output. default: "backward" See torch.fft.fft2
        pad_width: (tensor) pad width for the last spatial dimensions. 
        shift: flag for shifting the input data to make the zero-frequency located at the center of the arrc
    Returns:
        arr_c_fft: [B,Ch,R,C] FFT of the input complex tensor
    """

    if pad_width is not None:
        if padval == 0:
            arr_c = torch.nn.functional.pad(arr_c, pad_width)
        else:
            return NotImplementedError('zero padding is only implemented for now')

    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2,-1)) if shift else arr_c
    arr_c_shifted_fft_c = torch.fft.fft2(arr_c_shifted , norm=normalized)
    arr_c_fft = torch.fft.fftshift(arr_c_shifted_fft_c, dim=(-2,-1)) if shift else arr_c_shifted_fft_c

    return arr_c_fft



def ifft(arr_c, normalized="backward", pad_width=None, shift=True):
    """
    Compute the inverse Fast Fourier transform of a complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        normalized: Normalize the iFFT output. default: "backward" See torch.fft.ifft2
        pad_width: (tensor) pad width for the last spatial dimensions. 
        shift: flag for inversely shifting the input data 
    Returns:
        arr_c_fft: [B,Ch,R,C] inverse FFT of the input complex tensor
    """

    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2,-1)) if shift else arr_c
    arr_c_shifted_fft_c = torch.fft.ifft2(arr_c_shifted, norm=normalized)
    arr_c_fft = torch.fft.fftshift(arr_c_shifted_fft_c, dim=(-2,-1)) if shift else arr_c_shifted_fft_c

    if pad_width is not None:
        if pad_width[2] != 0 and pad_width[3] != 0:
            arr_c_fft = arr_c_fft[..., pad_width[2]:-pad_width[3], :]
        if pad_width[0] != 0 and pad_width[1] != 0:
            arr_c_fft = arr_c_fft[..., :, pad_width[0]:-pad_width[1]]

    return arr_c_fft


def conv_fft(img_c, kernel_c, pad_width=None):
    """
    Compute the convolution of an image with a convolution kernel using FFT
    Args:
        img_c: [B,Ch,R,C] image as a complex tensor 
        kernel_c: [B,Ch,R,C] convolution kernel as a complex tensor
        pad_width: (tensor) pad width for the last spatial dimensions. should be (0,0,0,0) for circular convolution. for linear convolution, pad zero by the size of the original image
    Returns:
        im_conv: [B,Ch,R,C] blurred image
    """

    img_fft = fft(img_c, pad_width=pad_width)
    kernel_fft = fft(kernel_c, pad_width=pad_width)
    return ifft( img_fft * kernel_fft, pad_width=pad_width)

def conv_fft_img_psf(img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """
    Compute the convolution of an image with a PSF (Point Spread Function) using FFT
    Args:
        img: [B, Ch, R, C] image tensor
        psf: [pB, 1, pR, pC] PSF tensor that is desired to be convolved with the image
    Returns:
        im_conv: [B, Ch, R, C] convolved image
    """
    B, Ch, R, C = img.shape
    pB, pCh, pR, pC = psf.shape
    assert (pB, pCh) == (B, 1)
    if Ch != 1:
        psf = torch.cat([psf for i in range(Ch)], axis=1)
    if (pR, pC) != (R, C):
        w1 = (R - pR) // 2
        w2 = R - pR - w1
        h1 = (C - pC) // 2
        h2 = C - pC - h1
        psf = F.pad(psf, (h1, h2, w1, w2))
        
    i = torch.fft.fftshift(torch.fft.fft2(img))
    p = torch.fft.fftshift(torch.fft.fft2(psf))
    
    im_conv = abs(torch.fft.ifftshift(torch.fft.ifft2(i * p)))
    
    return im_conv