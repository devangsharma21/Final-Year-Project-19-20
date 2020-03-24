# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:46:49 2020

@author: Devang
"""

from pydub import AudioSegment

import os 
path = os.getcwd() 
dir_list = os.listdir(path) 
print(len(dir_list))

#dir_list = ['a0001.wav']
c=1
for j in dir_list:
    
    audio = AudioSegment.from_wav(j)

    n = len(audio) 
    
    counter = 1
    interval = 5 * 1000
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
      
        chunk = audio[start:end] 
      
        filename = str(c)+'chunk'+str(counter)+'.wav'
      
        chunk.export(filename, format ="wav") 
        print("Processing chunk "+str(counter)+". Start = "
                            +str(start)+" end = "+str(end)) 
       
        counter = counter + 1
        
        if flag == 1:
            break
        
    os.remove(str(c)+"chunk"+str(counter-1)+".wav")
#    st = str(c)+"chunk1.wav"
#    pat = os.getcwd() 
#    dir_l = os.listdir(pat) 
#    if st in dir_l:
#        os.remove(str(c)+"chunk1.wav")
    c=c+1

