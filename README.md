# Music-Recommended system Using CNN

Using traditional image processing techinque to categorize audio into different song generes. The idea is to map input audio into a certain genere using a Convolutional Neural Network so that songs of the same class can be recommended to an user.

## 1. Dataset
- dataset on kaggle:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

- Dataset made and preprocessed to log-specturms by **XiplusChenyu**:
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
![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/melspecturm.png?raw=true)   ![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/logmelspecturm.png?raw=true)

## CNN Network
Originally I started with a really simple 3 layered CNN (2 convolutional layer and 1 output layer). However through rigorous testing it is clear that this simple architecture will always produce an result that is overfitted at around 50% accuracy. This can be seen in: https://github.com/wu2ming/Song-Classification/blob/main/song_classification_old.ipynb

At the end I was only able to reach at best 60% accuracy shown below:
