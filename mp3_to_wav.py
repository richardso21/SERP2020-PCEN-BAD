import os
from pydub import AudioSegment
from tqdm import tqdm

direct = '/scratch/richardso21/labeledresults'
os.chdir(direct)
lshort = os.listdir(direct)
lshort.sort()

#n = 0
for i in tqdm(lshort):
    load = AudioSegment.from_mp3(i)
    load = load.set_channels(1)
    wav = i[:-3] + 'wav'
    load.export(wav, format = 'wav')
    target = direct + '/' + i
    if os.path.isfile(target):
        os.unlink(target)
    #n += 1
    #print(int((n/len(lshort))*100),'% done')

#!cd {direct};rm *.mp3

