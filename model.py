'''
if prediceted value is more than 0.5 then survey taker is happy about the company 
'''
from nltk import word_tokenize
from keras.preprocessing import sequence
from keras.models import load_model
from keras.datasets import imdb
max_review_length = 1600
text_to_predict = "i am loving my work"

word2index = imdb.get_word_index()
test=[]
for word in word_tokenize(text_to_predict):
     test.append(word2index[word])
model = load_model("model.h5")
test=sequence.pad_sequences([test],maxlen=max_review_length)
prediction = model.predict(test)
print(prediction)
