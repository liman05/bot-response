#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# In[24]:


get_ipython().run_cell_magic('writefile', 'intends.json', '{\n    "intents": [\n        {\n            "tag": "greeting",\n            "inputs": ["hello", "hi", "hey"],\n            "responses": ["Hello! How can I assist you?", "Hi there!", "Hey, nice to see you!"]\n        },\n        {\n            "tag": "farewell",\n            "inputs": ["bye", "goodbye", "see you later"],\n            "responses": ["Goodbye! Take care.", "See you later!", "Farewell!"]\n        },\n        {\n            "tag": "thanks",\n            "inputs": ["thank you", "thanks"],\n            "responses": ["You\'re welcome!", "No problem.", "Glad I could help."]\n        },\n        {\n            "tag": "identify",\n            "inputs": ["who are you","what are you","whats your name", "what should i call you","hey am called name"],\n            "responses": ["i am a chatbot, call me blitzbot", "call me blitzbot, how about you", "i am blitzbot and am happy to talk to you"]\n        },\n        \n        {\n            "tag": "custom",\n            "inputs": ["Tell me a joke", "What\'s your favorite color?"],\n            "responses": ["Why did the scarecrow win an award? Because he was outstanding in his field!", "I don\'t have a favorite color, but I like all the colors!"]\n        }\n    ]\n}')


# In[25]:


with open('intends.json') as intend:
    d1 = json.load(intend)


# In[26]:


tags = []
inputs= []
responses  = {}
for intent in d1['intents']:
#     responses.append(intent['responses'])
    responses[intent['tag']]=intent['responses']
    for lines in intent['inputs']:
        inputs.append(lines)
        tags.append(intent['tag'])


# In[27]:


data = pd.DataFrame({"inputs":inputs, "tags":tags})
data


# In[31]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
#tokenizing data, appling padding and encoding output


# In[32]:


input_shape = x_train.shape[1]
print(input_shape)


# In[37]:


vocabulary = len(tokenizer.word_index)
print("number of unique words: ",vocabulary)
output_length = le.classes_.shape[0]
print("output lenght: ",output_length)


# In[38]:


i = Input(shape=(input_shape))
x = Embedding(vocabulary+1, 10)(i)
x= LSTM(10, return_sequences=True)(x)
x= Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i,x)


# In[44]:


model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


# In[ ]:


train = model.fit(x_train, y_train, epochs=200)
model.save('blitzbot_response')


# In[46]:


plt.plot(train.history['accuracy'],label='training set accuracy')
plt.plot(train.history['loss'],label='training set loss')
plt.legend()


# In[ ]:


import random
import string

while True:
    texts_p = []
    prediction_input = input('you: ')
    
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    #remove punctuatuion above oh
    
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)
    
    output= model.predict(prediction_input)
    output = output.argmax()
    #getting output from model
    
    response_tag = le.inverse_transform([output])[0]
    print("bot: ", random.choice(responses[response_tag]))
    if response_tag == "goodbye":
        break
    


# In[ ]:




