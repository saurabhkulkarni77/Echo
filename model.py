'''
Using this, from users answers related we can predict whether employee is happy or unhappy without explicitly asking by creating seperate question 
whether he is happy or not 

'''

from nltk import word_tokenize
from keras.preprocessing import sequence
from keras.models import load_model
from keras.datasets import imdb
max_review_length = 1600

word2index = imdb.get_word_index()
test=[]
for word in word_tokenize( "i am loving it"):
     test.append(word2index[word])
model = load_model("model.h5")
test=sequence.pad_sequences([test],maxlen=max_review_length)
prediction = model.predict(test)
print(prediction)
