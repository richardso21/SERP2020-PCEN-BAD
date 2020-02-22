#%%
import librosa
import librosa.display as display
import numpy
import IPython.display as ipd
#%%
old_audio = '/scratch/richardso21/mp3splt files/nigliq1/NIGLIQ_short_test/NIGLIQ1_20160607_203214_2960m_00s__2980m_00s_17m_40s__17m_50s.wav'
#%%
old,sr = librosa.load(old_audio)
ipd.Audio(old_audio)
#%%
Old_db = librosa.power_to_db(librosa.feature.melspectrogram(old,sr=sr))
display.specshow(Old_db,y_axis='hz')

#%%
mel = librosa.feature.melspectrogram(old,sr=sr)
new = librosa.pcen(mel)
display.specshow(new,y_axis='hz')

#%%
import soundfile as sf
sf.write('pcen_test.wav',new,samplerate=sr)
ipd.Audio('pcen_test.wav')

#%%
