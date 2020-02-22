#!/usr/bin/env python
# coding: utf-8

# In[54]:


#pip install tqdm


# In[1]:


import librosa, h5py, os, multiprocessing
import librosa.display as display
import numpy as np
import IPython.display as ipd
from tqdm import tqdm


# In[18]:


PCEN = True # True or False


# In[19]:


base_dir = '/scratch/richardso21/DCASE_3_Models/grill_pcen'
os.chdir(base_dir)


# ---

# In[20]:


def wav_to_h5(input_wav_dir):
    a,sr = librosa.load(input_wav_dir)
    
    a_log = librosa.feature.melspectrogram(a,sr=sr,n_fft=1024,hop_length=315,n_mels=80,fmax=11000,power=1)
    if PCEN == True:
        a_out = librosa.pcen(a_log*(2**31))
    elif PCEN == False:
        a_out = librosa.power_to_db(a_log,ref=np.max)
    
    duration = librosa.get_duration(a)
    frames = a_out.shape[1]
    timeframe = np.linspace(0,duration,num=frames)
    
    dir_name = os.path.basename(os.path.dirname(input_wav_dir))
    name = os.path.basename(input_wav_dir)
    
    with h5py.File('workingfiles/spect/{0}/{1}.h5'.format(dir_name,name), 'w') as data_file:
        data_file.create_dataset('features',data=a_out.T,dtype='float32')
        data_file.create_dataset('times',data=timeframe,dtype='float32')
    #print('Done creating {0}'.format(name))


# In[ ]:


for folder in os.listdir('audio'):
    filelist_path = os.path.join('audio',folder)
    filelist = sorted(os.listdir(filelist_path))
    
    try:
        os.mkdir(os.path.join('workingfiles/spect',folder))
    except OSError as error:
        print(error)
        
    for file in tqdm(filelist,desc=folder):
        file_path = os.path.join('workingfiles/spect',folder,file)
            
        if os.path.exists('{0}.h5'.format(file_path)) == False:
            #p = multiprocessing.Process(target=wav_to_h5,args=(os.path.join(filelist_path,file),))
            #p.start() #now uses multiple processors
            #p.join()
            wav_to_h5(os.path.join(filelist_path,file))
        else:
            break


# In[ ]:




