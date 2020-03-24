# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:50:09 2020

@author: Devang
"""

from pydub import AudioSegment

import os 
path = os.getcwd() 
dir_list = os.listdir(path) 
#print(dir_list)
s=0
l=[]
for j in dir_list:
    
    audio = AudioSegment.from_wav(j)

    n = len(audio)
    l.append(n)
    s+=n
print((s/len(dir_list)/1000))
print(max(l)/1000)
print(min(l)/1000)

