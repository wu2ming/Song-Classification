# Music-Recommend system Using CNN

Using traditional image processing techinque to categorize audio into different song genres. The idea is to map input audio into a certain genre using a Convolutional Neural Network so that songs of the same class can be recommended to an user.

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
    
'''
#logmelspecturm
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
'''

** It could Look like this **
![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/melspecturm.png?raw=true)   ![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/logmelspecturm.png?raw=true)

## CNN Network
Originally I started with a really simple 3 layered CNN (2 convolutional layer and 1 output layer). However through rigorous testing it is clear that this simple architecture will always produce an result that is overfitted at around 50% accuracy. This can be seen in: https://github.com/wu2ming/Song-Classification/blob/main/song_classification_old.ipynb

At the end I was only able to reach at best 60% accuracy shown below:
![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/overfit.png?raw=true)

![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/cnn_1_test.png?raw=true)   

## CRNN Network
This network architecture was the idea from this article by XiplusChenyu (https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/music_genre_classification.pdf).  
![alt_text](https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/pictures/crnn.png)
The network consists of 3 convolution layer, a GRU layer and 3 fully-connected layer. Through the use of a more complex architecture along with neuron regularization and batch normalization, the Deep Network is able to pickout the semantics of the audio spectrum unlike in version1 of our CNN. The network is not memorizing the training set anymore, and this can be evident in out final test result
![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/crnn_test.png?raw=true)
    
    
    - We can easily see that although overfitting still occurs at the end, our validation data is much more aligned with our training data.

At the end we got around a 80% accuracy with our imporved network
![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/test_final.png?raw=true)
![alt text](https://github.com/wu2ming/Song-Classification/blob/main/images/matrix.png?raw=true)

## Future
As we can see from our confusion matrix, the network is familair with some of genres but struggles to identity other ones in this context. However, audio is just one aspect of music. We can still lyrics, chords, instruments, producers, artists to categorize music in an more diverse way.
