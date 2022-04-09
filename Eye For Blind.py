#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
files=[]
import os
for dirname, _, filenames in os.walk('/kaggle/input'):    
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
    print(files)


# In[3]:


get_ipython().system(' pip install gTTS')


# In[4]:


#Import all the required libraries
#System libraries
import os, time
from tqdm import tqdm
import glob
# Data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# Model building 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#Read/Display images
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# import tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
from IPython import display
# Import the required module for text to speech conversion
from gtts import gTTS


# In[5]:


#Import the dataset and read the image into a seperate variable
images='D:\\flicker\\Images' # importing images from input path
all_imgs = glob.glob(images + '/*.jpg',recursive=True) 
print("The total images present in the dataset: {}".format(len(all_imgs)))


# In[6]:


#Visualise both the images & text present in the dataset
all_imgs


# In[7]:


#Import the dataset and read the text file into a seperate variable
text_file = "D:\\flicker\\captions.txt" #storing captions in a separate variable
def load_doc(filename):    
    file = open(filename)
    text = file.read()
    file.close()
    return text 
doc = load_doc(text_file)
print(doc[:300])


# In[8]:


img = 'D:\\flicker\\Images'
def get_img_ids_and_captions(text):
    keys=[]
    values=[]
    key_paths=[]
    text=text.splitlines()[1:]
    for line in text:
        com_idx=line.index(",")
        im_id,im_cap=line[:com_idx],line[com_idx+1:]
        keys.append(im_id)
        values.append(im_cap)
        key_paths.append(img+'/'+im_id)
    return keys,key_paths, values


# In[9]:


all_img_id, all_img_vector, annotations = get_img_ids_and_captions(doc)
df = pd.DataFrame(list(zip(all_img_id, all_img_vector,annotations)),columns =['ID','Path', 'Captions']) 
df.head(10)


# In[10]:


len (annotations)


# In[11]:


type (annotations)


# In[12]:


#check total captions and images present in dataset
print("Total captions present in the dataset: "+ str(len(annotations)))
print("Total images present in the dataset: " + str(len(all_imgs)))


# In[13]:


df.info()


# In[14]:


def plot_image_captions(Pathlist,captionsList,fig,count=2,npix=299,nimg=2):
        image_load = load_img(Path,target_size=(npix,npix,3))
        ax = fig.add_subplot(nimg,2,count,xticks=[],yticks=[])
        ax.imshow(image_load)
        count +=1
        ax = fig.add_subplot(nimg,2,count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(0,len(captions))
        for i, caption in enumerate(captions):
            ax.text(0,i,caption,fontsize=20)


# In[15]:


#Images and captions
fig = plt.figure(figsize=(10,20))
count = 1
for Path in df[:20].Path.unique():
    captions = list(df["Captions"].loc[df.Path== Path].values)
    plot_image_captions(Path,captions,fig,count,299,5)
    count +=2
plt.show()


# In[16]:


print(df['Captions'])


# In[17]:


#Create a list which contains all the captions
annotations=df.Captions.apply(lambda z:"<start>"+" "+z+" "+"<end>")
#Create a list which contains all the path to the images
all_img_path=list(df["Path"])
print("Total captions present in the dataset: "+ str(len(annotations)))
print("Total images present in the dataset: " + str(len(all_img_path)))


# In[18]:


#Create the vocabulary & the counter for the captions
def create_vocabulary(data):
  vocab = []
  for captions in data.Captions.values:
    vocab.extend(captions.split())
  print("Vocabulary Size : {}".format(len(set(vocab))))
  return vocab


# In[19]:


from collections import Counter
vocabulary = create_vocabulary(df)#write your code here
val_count=Counter(vocabulary)
val_count


# In[20]:


#Visualise the top 30 occuring words in the captions
top30= val_count.most_common(30) # storing Top30 words in a variable
top30


# In[21]:


df_word = pd.DataFrame.from_dict(val_count, orient = 'index')
df_word = df_word.sort_values(by = [0], ascending=False).reset_index()
df_word = df_word.rename(columns={'index':'word', 0:'count'})
df_word.head()


# In[22]:


words = []
counts = []
for word_count in top30 : 
    words.append(word_count[0])
    counts.append(word_count[1])
plt.figure(figsize=(20,10))
plt.title('Top 30 words in the vocabulary \n', color='b',size= 20)
plt.xlabel('Words',size=15)
plt.ylabel('Count', size=15)
plot = sns.barplot(x=words, y=counts, color='orange')
for p in plot.patches:
    plot.annotate(format(int(p.get_height())), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.show()


# In[23]:


#Create the tokenized vectors by tokenizing the captions.This gives us a vocabulary of all of the unique words in the data. Keep the total vocaublary to top 5,000 words for saving memory.
#Replacing all other words with the unknown token "UNK" .
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# Create word-to-index and index-to-word mappings
tokenizer.fit_on_texts(annotations)
#transform each text into a sequence of integers
tokensized_text = tokenizer.texts_to_sequences(annotations)


# In[24]:


tokensized_text[:5] #first 5 instance


# In[25]:


annotations[:5]


# In[26]:


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
tokensized_text = tokenizer.texts_to_sequences(annotations)


# In[27]:


tokenizer.word_counts


# In[28]:


tokenizer.index_word


# In[29]:


print(tokenizer.oov_token)
print(tokenizer.index_word[0])


# In[30]:


# Create a word count of your tokenizer to visulize the Top 30 occuring words after text processing
tokenizer_df = pd.DataFrame([tokenizer.word_counts]).transpose().reset_index()
top30words = tokenizer_df.sort_values(by=0,ascending=False).head(30).reset_index(drop=True).rename(columns={"index":"words",0:"counts"})
plt.figure(figsize=(20,10))
plt.title('Top 30 occuring words in the vocabulary after text processing \n',color='b')
plt.xlabel('Words')
plt.ylabel('Count')
plot=sns.barplot(x=top30words.words,y=top30words.counts, color='lightblue')
for p in plot.patches:
    plot.annotate(format(int(p.get_height())), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.show()


# In[31]:


# Pad each vector to the max_length of the captions ^ store it to a vairable
def max_len(input):
    listofLength = [len(x) for x in input]
    return max(listofLength)
max_l = max_len(tokensized_text) # storing all lengths in list.Python list method max returns the elements from the list with maximum value.
#calculate longest word_length and pads all sequences to equal length as that of the longest
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(tokensized_text,padding='post',maxlen=max_l)
print("The shape of Caption vector is :" + str(cap_vector.shape))


# In[32]:


cap_vector


# In[33]:


create the dataset consisting of image paths
all_imgs


# In[34]:


#creating the function which returns images & their path
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# In[35]:


#applying the function to the image path dataset, such that the transformed dataset should contain images & their path
plt.imshow(load_image(all_imgs[4])[0])
plt.show()


# In[36]:


# sort the unique paths and store in a list
encode_train_set = sorted(set(all_img_vector))
feature_dict = {}
image_data_set = tf.data.Dataset.from_tensor_slices(encode_train_set)
image_data_set = image_data_set.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)


# In[37]:


image_data_set


# In[38]:


image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input =image_model.input
hidden_layer =  image_model.layers[-1].output #to get the output of the image_model
image_features_extract_model =  keras.Model(new_input, hidden_layer)  #build the final model using both input & output layer


# In[39]:


#apply the feature_extraction model to your earlier created dataset which contained images & their respective paths
#Once the features are created, you need to reshape them such that feature shape is in order of (batch_size, 8*8, 2048)
image_features_extract_model.summary()


# In[40]:


# we are using tqdm for progress bar
for image,path in tqdm(image_data_set): 
    features = image_features_extract_model(image)# feed images from newly created Dataset above to Inception V3 built above
    features = tf.reshape(features,(features.shape[0],-1,features.shape[3])) # features in a batch
    for batch_features, p in zip(features, path):
        path_of_feature = p.numpy().decode("utf-8")
        feature_dict[path_of_feature] =  batch_features.numpy()


# In[41]:


batch_features.shape


# In[42]:


path_train, path_test, cap_train, cap_test = train_test_split(all_img_vector,
                                                                    cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=42)


# In[43]:


print("Training data for images: " + str(len(path_train)))
print("Testing data for images: " + str(len(path_test)))
print("Training data for Captions: " + str(len(cap_train)))
print("Testing data for Captions: " + str(len(cap_test)))


# In[44]:


# Create a function which maps the image path to their feature. 
# This function will take the image_path & caption and return it's feature & respective caption.
def map_func(image_name,capt):
  img_tensor = feature_dict[image_name.decode('utf-8')]
  return img_tensor, capt


# In[45]:


# create a builder function to create dataset which takes in the image path & captions as input
# This function should transform the created dataset(img_path,cap) to (features,cap) using the map_func created earlier
def gen_dataset(images_data, captions_data, BATCH_SIZE =32):    
    dataset = tf.data.Dataset.from_tensor_slices((images_data, captions_data)) # dataset created using tf.data.Dataset.from_tensor_slices
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
# .prefetch() is used to prepare all upcoming elements, while current elements are being processed
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


# In[46]:


train_dataset=gen_dataset(path_train,cap_train)
test_dataset=gen_dataset(path_test,cap_test)


# In[47]:


train_dataset


# In[48]:


test_dataset


# In[49]:


sample_img_batch, sample_cap_batch = next(iter(train_dataset))
print(sample_img_batch.shape)  #(batch_size, 8*8, 2048)
print(sample_cap_batch.shape) #(batch_size,max_len)


# In[50]:


embedding_dim = 256 
units = 512
vocab_size = 5001 #top 5,000 words +1
train_num_steps = len(path_train) #len(total train images) // BATCH_SIZE
test_num_steps = len(path_test) #len(total test images) // BATCH_SIZE


# In[51]:


train_num_steps


# In[52]:


test_num_steps


# In[53]:


#Building Encoder using CNN Keras subclassing method
class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim) #build your Dense layer with relu activation
    def call(self, features):
        features = self.dense(features)# extract the features from the image shape: (batch, 8*8, embed_dim)
        features = tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0) 
        return features


# In[54]:


encoder=Encoder(embedding_dim)


# In[55]:


encoder


# In[56]:


class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) #build your Dense layer
        self.W2 = tf.keras.layers.Dense(units)#build your Dense layer
        self.V = tf.keras.layers.Dense(1)#build your final Dense layer with unit 1
        self.units=units
    def call(self, features, hidden):
        #features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        hidden_with_time_axis =  tf.expand_dims(hidden, 1)# Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))# build your score funciton to shape: (batch_size, 8*8, units)
        attention_weights =  tf.keras.activations.softmax(self.V(score), axis=1)# extract your attention weights with shape: (batch_size, 8*8, 1)
        context_vector = attention_weights * features #shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1) # reduce the shape to (batch_size, embedding_dim)
        return context_vector, attention_weights


# In[57]:


class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units) #iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer
    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed = self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1) # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        return output,state, attention_weights
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# In[58]:


decoder=Decoder(embedding_dim, units, vocab_size)


# In[59]:


features=encoder(sample_img_batch)
hidden = decoder.init_state(batch_size=sample_cap_batch.shape[0])
dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * sample_cap_batch.shape[0], 1)
predictions, hidden_out, attention_weights= decoder(dec_input, features, hidden)
print('Feature shape from Encoder: {}'.format(features.shape)) #(batch, 8*8, embed_dim)
print('Predcitions shape from Decoder: {}'.format(predictions.shape)) #(batch,vocab_size)
print('Attention weights shape from Decoder: {}'.format(attention_weights.shape)) #(batch, 8*8, embed_dim)


# In[60]:


optimizer =keras.optimizers.Adam() #define the optimizer
loss_object =keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #define your loss object


# In[61]:


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# In[62]:


checkpoint_path = "./Checkpoints"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[63]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])


# In[64]:


@tf.function
def train_step(img_tensor, target):
    loss = 0
    hidden = decoder.init_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)
        avg_loss = (loss/int(target.shape[1]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, avg_loss


# In[65]:


@tf.function
def test_step(img_tensor, target):
    loss = 0
    hidden = decoder.init_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)
        avg_loss = (loss / int(target.shape[1]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, avg_loss


# In[66]:


def test_loss_cal(test_dataset):
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(test_dataset):
        batch_loss, t_loss = test_step(img_tensor, target)
        total_loss += t_loss
    avg_test_loss=total_loss/test_num_steps
    #to get the average loss result on your test data    
    return avg_test_loss


# In[ ]:


loss_plot = []
test_loss_plot = []
EPOCHS = 15
best_test_loss=100
for epoch in tqdm(range(0, EPOCHS)):
    start = time.time()
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(train_dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        avg_train_loss=total_loss / train_num_steps
    loss_plot.append(avg_train_loss)    
    test_loss = test_loss_cal(test_dataset)
    test_loss_plot.append(test_loss)
    print ('For epoch: {}, the train loss is {:.3f}, & test loss is {:.3f}'.format(epoch+1,avg_train_loss,test_loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    if test_loss < best_test_loss:
        print('Test loss has been reduced from %.3f to %.3f' % (best_test_loss, test_loss))
        best_test_loss = test_loss
        ckpt_manager.save()


# In[ ]:


from matplotlib.pyplot import figure
figure(figsize=(12, 8))
# plt.plot(loss_plot)
# plt.plot(test_loss_plot)
plt.plot(loss_plot, color='blue', label = 'training_loss_plot')
plt.plot(test_loss_plot, color='orange', label = 'test_loss_plot')
plt.xlabel('Epochs',fontsize = 15, color='green')
plt.ylabel('Loss',fontsize = 15, color='green')
plt.title('Loss Plot',fontsize = 20,color='Red')
plt.legend()
plt.show()


# In[ ]:


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.init_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0) #process the input image to desired format before extracting features
    img_tensor_val = image_features_extract_model(temp_input)# Extract features using our feature extraction model
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)# extract the features by passing the input to encoder
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(max_length):
        predictions, hidden, attention_weights =decoder(dec_input, features, hidden)
 # get the output from decoder
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id =tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id]) #extract the predicted id(embedded value) which carries the max value
        #map the id to the word from tokenizer and append the value to the result list
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot,predictions
        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot,predictions


# In[ ]:


def beam_evaluate(image, beam_index =1):
    max_length=max_l
    start = [tokenizer.word_index['<start>']]
    result = [[start, 0.0]]
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.init_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    while len(result[0][0]) < max_length:
        i=0
        temp = []
        for s in result:
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            i=i+1
            word_preds = np.argsort(predictions[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += np.log(predictions[0][w])
                temp.append([next_cap, prob])
        result = temp
        result = sorted(result, reverse=False, key=lambda l: l[1])
        result = result[-beam_index:]
        predicted_id = result[-1]
        pred_list = predicted_id[0]
        prd_id = pred_list[-1] 
        if(prd_id!=3):
            dec_input = tf.expand_dims([prd_id], 0)  
        else:
            break
    result2 = result[-1][0]
    intermediate_caption = [tokenizer.index_word[i] for i in result2]
    final_caption = []
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    attention_plot = attention_plot[:len(result), :]
    final_caption = ' '.join(final_caption[1:])
    return final_caption


# In[ ]:


def plot_attmap(caption, weights, image):
    fig = plt.figure(figsize=(10, 10))
    temp_img = np.array(Image.open(image))
    len_cap = len(caption)
    for cap in range(len_cap):
        weights_img = np.reshape(weights[cap], (8,8))
        weights_img = np.array(Image.fromarray(weights_img).resize((224, 224), Image.LANCZOS))
        ax = fig.add_subplot(len_cap//2, len_cap//2, cap+1)
        ax.set_title(caption[cap], fontsize=15)
        img=ax.imshow(temp_img)
        ax.imshow(weights_img, cmap='gist_heat', alpha=0.6,extent=img.get_extent())
        ax.axis('off')
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu


# In[ ]:


def filt_text(text):
    filt=['<start>','<unk>','<end>'] 
    temp= text.split()
    [temp.remove(j) for k in filt for j in temp if k==j]
    text=' '.join(temp)
    return text


# In[ ]:


features_shape = batch_features.shape[1]
attention_features_shape = batch_features.shape[0]


# In[ ]:


rid = np.random.randint(0, len(path_test))
max_length=max_l
test_image = path_test[rid]
#test_image = './images/413231421_43833a11f5.jpg'
#real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
result, attention_plot,pred_test = evaluate(test_image)
real_caption=filt_text(real_caption)
pred_caption=' '.join(result).rsplit(' ', 1)[0]
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption.split()
score = sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25) )#set your weights
print(f"BLEU score: {score*100}")
print('Real Caption:', real_caption)
print('Prediction Caption:', pred_caption)
plot_attmap(result, attention_plot, test_image)
Image.open(test_image)


# In[ ]:


captions=beam_evaluate(test_image)
print(captions)


# In[ ]:


# Import the required module for text to speech conversion
from gtts import gTTS
# Language in which you want to convert
language = 'en'
# Passing the text and language to the engine, 
myobj = gTTS(text=pred_caption, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
audio_file_name = "Predicted_text.mp3"
# Playing the converted file
myobj.save(audio_file_name)
display.display(display.Audio(audio_file_name, rate=None,autoplay=False)) # Playing the saved file


# In[ ]:


rid = np.random.randint(0, len(path_test))
max_length=max_l
test_image = path_test[rid]
#test_image = './images/413231421_43833a11f5.jpg'
#real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
result, attention_plot,pred_test = evaluate(test_image)
real_caption=filt_text(real_caption)      
pred_caption1=' '.join(result).rsplit(' ', 1)[0]
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption1.split()
score = sentence_bleu(reference, candidate, weights=(0.35,0.25,0.25,0) )#set your weights
print(f"BLEU score: {score*100}")
print('Real Caption:', real_caption)
print('Prediction Caption:', pred_caption1)
plot_attmap(result, attention_plot, test_image)
Image.open(test_image)


# In[ ]:


captions1=beam_evaluate(test_image)
print(captions1)


# In[ ]:


# Language in which you want to convert
language = 'en'
# Passing the text and language to the engine, 
myobj1 = gTTS(text=pred_caption1, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
audio_file_name1 = "v1.mp3"
# Playing the converted file
myobj1.save(audio_file_name1)
display.display(display.Audio(audio_file_name1, rate=None,autoplay=False)) # Playing the saved file


# In[ ]:


rid = np.random.randint(0, len(path_test))
max_length=max_l
test_image = path_test[rid]
#test_image = './images/413231421_43833a11f5.jpg'
#real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
result, attention_plot,pred_test = evaluate(test_image)
real_caption=filt_text(real_caption)      
pred_caption2=' '.join(result).rsplit(' ', 1)[0]
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption2.split()
score = sentence_bleu(reference, candidate, weights=(0.5,0.25,0,0) )#set your weights)
print(f"BLEU score: {score*100}")
print('Real Caption:', real_caption)
print('Prediction Caption:', pred_caption2)
plot_attmap(result, attention_plot, test_image)
Image.open(test_image)


# In[ ]:


captions2=beam_evaluate(test_image)
print(captions2)


# In[ ]:


# Language in which you want to convert
language = 'en'
# Passing the text and language to the engine, 
myobj2 = gTTS(text=pred_caption2, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
audio_file_name2 = "v2.mp3"
# Playing the converted file
myobj2.save(audio_file_name2)
display.display(display.Audio(audio_file_name2, rate=None,autoplay=False)) # Playing the saved file


# In[ ]:


rid = np.random.randint(0, len(path_test))
max_length=max_l
test_image = path_test[rid]
#test_image = './images/413231421_43833a11f5.jpg'
#real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
result, attention_plot,pred_test = evaluate(test_image)
real_caption=filt_text(real_caption)      
pred_caption3=' '.join(result).rsplit(' ', 1)[0]
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption3.split()
score = sentence_bleu(reference, candidate, weights=(0.35,0.35,0,0) )#set your weights)
print(f"BLEU score: {score*100}")
print('Real Caption:', real_caption)
print('Prediction Caption:', pred_caption3)
plot_attmap(result, attention_plot, test_image)
Image.open(test_image)


# In[ ]:


captions3=beam_evaluate(test_image)
print(captions3)


# In[ ]:


# Language in which you want to convert
language = 'en'
# Passing the text and language to the engine, 
myobj3 = gTTS(text=pred_caption3, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
audio_file_name3 = "v3.mp3"
# Playing the converted file
myobj3.save(audio_file_name3)
display.display(display.Audio(audio_file_name3, rate=None,autoplay=False)) # Playing the saved file


# In[ ]:


rid = np.random.randint(0, len(path_test))
max_length=max_l
test_image = path_test[rid]
#test_image = './images/413231421_43833a11f5.jpg'
#real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
result, attention_plot,pred_test = evaluate(test_image)
real_caption=filt_text(real_caption)      
pred_caption4=' '.join(result).rsplit(' ', 1)[0]
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption4.split()
score = sentence_bleu(reference, candidate, weights=(0.5,0.5,0,0) )#set your weights)
print(f"BLEU score: {score*100}")
print('Real Caption:', real_caption)
print('Prediction Caption:', pred_caption4)
plot_attmap(result, attention_plot, test_image)
Image.open(test_image)


# In[ ]:


captions4 =beam_evaluate(test_image)
print(captions4)


# In[ ]:


# Language in which you want to convert
language = 'en'
# Passing the text and language to the engine, 
myobj4 = gTTS(text=pred_caption4, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
audio_file_name4 = "v4.mp3"
# Playing the converted file
myobj4.save(audio_file_name4)
display.display(display.Audio(audio_file_name4, rate=None,autoplay=False)) # Playing the saved file


# In[ ]:


rid = np.random.randint(0, len(path_test))
max_length=max_l
test_image = path_test[rid]
#test_image = './images/413231421_43833a11f5.jpg'
#real_caption = '<start> black dog is digging in the snow <end>'
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
result, attention_plot,pred_test = evaluate(test_image)
real_caption=filt_text(real_caption)      
pred_caption5=' '.join(result).rsplit(' ', 1)[0]
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = pred_caption5.split()
score = sentence_bleu(reference, candidate, weights=(0.25,0.25,0,0) )#set your weights)
print(f"BLEU score: {score*100}")
print('Real Caption:', real_caption)
print('Prediction Caption:', pred_caption5)
plot_attmap(result, attention_plot, test_image)
Image.open(test_image)


# In[ ]:


captions5=beam_evaluate(test_image)
print(captions5)


# In[ ]:


# Language in which you want to convert
language = 'en'
# Passing the text and language to the engine, 
myobj5 = gTTS(text=pred_caption5, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
audio_file_name5 = "v5.mp3"
# Playing the converted file
myobj5.save(audio_file_name5)
display.display(display.Audio(audio_file_name5, rate=None,autoplay=False)) # Playing the saved file

