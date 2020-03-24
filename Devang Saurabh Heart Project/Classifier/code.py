# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:36:06 2020

@author: Devang
"""

import glob
import os
from pydub import AudioSegment
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image as pil_image

# ---------- Identify the latest wav file in the folder -----------------------
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
    for img in spects:
        test_image = image.load_img(j, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
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
    
else:
    display(disp=1)
    


