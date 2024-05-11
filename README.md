# Music-Recommended system Using CNN

Using traditional image processing techinque to categorize audio into different song generes. The idea is to map input audio into a certain genere using a Convolutional Neural Network so that songs of the same class can be recommended to an user.

## 1. Dataset
- dataset on kaggle
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

- Dataset made and preprocessed to log-specturms by **XiplusChenyu**
https://github.com/XiplusChenyu/Musical-Genre-Classification

### 1.1 Preprocess Data
The general way to turn audio into an image its to turn it into a specturm. The `librosa` library can used to easily turn audio into melspecturm or logmelspectrum
'''
#melspecturm in librosa
signal, sr = librosa.load(
    os.path.join(genre_folder, song))
melspec = librosa.feature.melspectrogram(
    signal, sr=sr).T[:1280, ]
    
#logmelspecturm
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
'''
** It could Look like this **

