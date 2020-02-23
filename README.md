# SERP Project Bird Audio Detection (BAD) with novel PCEN audio filter
This github repository is a complement to the presenting poster at the NYCSEF 2020 fair. This repository includes all source code of 
implementing PCEN onto Thomas Grill's [Bulbul Deep Learning Model](https://github.com/DCASE-REPO/bulbul_bird_detection_dcase2018), as well as the data analysis of results and graph production. It had 
achieved a preview AUC (area under the curve, see poster for details) score of .885 in the 
[DCASE 2018 Bird Audio Detection Challenge](http://dcase.community/challenge2018/task-bird-audio-detection-results). 

## Results
Application of PCEN was significantly beneficial to the model's performance, from a mean AUC score (out of 5 trials) of .848 to a .904 in this experiment.
|         | Trial 1  | Trial 2  | Trial 3  | Trial 4  | Trial 5  | Average  | P-Value  |
|---------|----------|----------|----------|----------|----------|----------|----------|
| no pcen | 0.859158 | 0.854338 | 0.820336 | 0.880204 | 0.826951 | _0.848197_ |          |
| pcen    | 0.914403 | 0.901837 | 0.899928 | 0.901896 | 0.903068 | _0.904226_ |          |
| T-Test  |          |          |          |          |          |          | __0.001097__ |

## Prerequisites for code
All components of the project is run on Python 3 (version should not make a difference). Packages used include:
* Pydub
* Librosa
* tqdm
* h5py
* Anaconda
  * Numpy
  * Pandas
  * SciPy
  * Scikit-Learn
* _For prerequisites in running the Bulbul model, look [here](https://github.com/DCASE-REPO/bulbul_bird_detection_dcase2018)._

## Acknowledgements
I would like to thank Dr. Michael I Mandel from Brooklyn
College CUNY as well as Dr. John Davis from Staten Island
Technical High School for assisting, advising, and supervising
me throughout my project.
