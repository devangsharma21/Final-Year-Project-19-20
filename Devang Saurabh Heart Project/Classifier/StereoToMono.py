# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:15:39 2020

@author: Devang
"""

from pydub import AudioSegment
sound = AudioSegment.from_wav("1chunk2.wav")
sound = sound.set_channels(1)
sound.export("mono.wav", format="wav")