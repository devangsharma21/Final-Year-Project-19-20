# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:15:41 2020

@author: Devang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from pydub import AudioSegment
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from keras.models import load_model
from keras.preprocessing import image

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None
    
if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
        

def load_img(pathi, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    img = pil_image.open(pathi)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x



list_of_files = glob.glob('*.wav')
latest_file = max(list_of_files, key=os.path.getctime)
# print (latest_file)
# -----------------------------------------------------------------------------


# ---------- Convert Audio from Stereo to Mono --------------------------------
sound = AudioSegment.from_wav(latest_file)
sound = sound.set_channels(1)
# sound.export("mono.wav", format="wav")
# -----------------------------------------------------------------------------


# ---------- Split Audio into smaller Segments---------------------------------
def audioSplit():
    n = len(sound) 
    
    if n<=4000:
        return 1
    
    else:
        
        if n<=21000 and n>4000:
            interval = 3 * 1000
            
        elif n>21000:
            interval = 8 * 1000
    
        counter = 1
        overlap = 0 * 1000
        start = 1 * 1000
        end = 0
        flag = 0
          
        for i in range(0, 2 * n, interval): 
              
            if i == 0: 
                start = 1 * 1000
                end = start + interval 
           
            else: 
                start = end - overlap 
                end = start + interval  
           
            if end >= n: 
                end = n 
                flag = 1
          
            chunk = sound[start:end] 
          
            filename ='segments/chunk'+str(counter)+'.wav'
          
            chunk.export(filename, format ="wav") 
            #print("Processing chunk "+str(counter)+". Start = "
                                #+str(start)+" end = "+str(end)) 
           
            counter = counter + 1
            
            if flag == 1:
                break
            
        os.remove("segments/chunk"+str(counter-1)+".wav")
        #    st = str(c)+"chunk1.wav"
        #    pat = os.getcwd() 
        #    dir_l = os.listdir(pat) 
        #    if st in dir_l:
        #        os.remove(str(c)+"chunk1.wav")
        return 0
#------------------------------------------------------------------------------


# ----------- Spectrogram Conversion ------------------------------------------
""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)
    
    #print(timebins)
    #print(freqbins)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    
    #print(samplerate)
    #print(samples)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=20., sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    #print("timebins: ", timebins)
    #print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.colorbar()

#    plt.xlabel("time (s)")
#    plt.ylabel("frequency (hz)")
#    plt.xlim([0, timebins-1])
#    plt.ylim([0, freqbins])
#
#    xlocs = np.float32(np.linspace(0, timebins-1, 5))
#    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
#    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
#    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return ims
# -----------------------------------------------------------------------------


# ------------------ Testing Spectrogram using CNN ----------------------------
def modelEval():
    model = load_model('G:/Heart Project/Dataset/RealTime/models/UMichGoodSpects.h5')
    spects = os.listdir(path+"/spectrograms")
    tot=0
    #print(spects)
    for im in spects:
        im="G:/Heart Project/Dataset/RealTime/spectrograms/"+im
        test_image = load_img(im, target_size = (64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
    
        result = model.predict(test_image)
        tot+=result
    avg=tot/len(spects)
    #print(result)
    res=1
    if avg<0.5:
        res=0
    # res = np.argmax(result)
    #print(tc[res])
    
    return res 
# -----------------------------------------------------------------------------
    

# ---------------- Display Results --------------------------------------------
def display(disp):
    if disp==1:
        print("Too Small")
        
    else:
        res = modelEval()
        tc = {0:'Murmur', 1: 'Normal'}
        print(tc[res])
# -----------------------------------------------------------------------------


# ------------------------ Delete Spectrograms and Segments -------------------
def del_files():
    pat = os.getcwd()
    die = os.listdir(path+"/segments")
    for seg in die:
        os.remove(path+"/segments/"+seg)
    fol = os.listdir(path+"/spectrograms")
    for seg in fol:
        os.remove(path+"/spectrograms/"+seg)
# -----------------------------------------------------------------------------


stop = audioSplit()
if stop==0:
    path = os.getcwd() 
    dir_list = os.listdir(path+"/segments") 
    # print(dir_list)
    
    c=1
    for j in dir_list:
        j="G:/Heart Project/Dataset/RealTime/segments/"+j
        ims = plotstft(audiopath = j, plotpath="G:/Heart Project/Dataset/RealTime/spectrograms/"+str(c)+".png")
        c+=1
    display(disp=0)
    del_files()
    
else:
    display(disp=1)
    

    