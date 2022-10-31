#%%
from tensorflow import keras
from keras import models,layers
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
import gdown
import os
from tqdm import tqdm

BATCH_SIZE = 64
LATENT_DIM = 100
IMG_SHAPE = (64,64,3)

x_train = keras.utils.image_dataset_from_directory('archive/images',image_size=(64,64),batch_size=BATCH_SIZE,shuffle=True,seed=123,labels=None)

#Normalisation des données
x_train = x_train.map(lambda x: (x-127.5)/127.5)
#%%
test_batch = next(iter(x_train))
fig = plt.figure(figsize=(12,12))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.axis('off')
  #On oublie pas de faire image * 0.5 + 0.5 pour revenir dans [0,1]
  plt.imshow(test_batch[i]*0.5+0.5)
plt.show()
#%%
def define_discriminator(im_shape=(64,64,3)):
  model = models.Sequential()
  ###Ajouter les couches ici (rappel: on peut utiliser model.add(layers.CoucheX(arguments)))
  model.add(layers.Conv2D(32,3, strides=2, padding='same',input_shape=im_shape))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Conv2D(64,3, strides=2, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Conv2D(128,3, strides=2, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Conv2D(256,3, strides=2, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Conv2D(1,4,1,padding='valid'))
  model.add(layers.Flatten())  
  return model

##Petit tips : faire un .summary() pour vérifier qu'on s'est pas trompé dans les dimensions de sortie
define_discriminator().summary()

def define_generator(latent_dim=LATENT_DIM):
	model = models.Sequential()
	###Ajouter les couches ici
	model.add(layers.Dense(4*4*1024,input_shape=(latent_dim,)))
	model.add(layers.Reshape((4,4,1024)))
	model.add(layers.Conv2DTranspose(256,3,2,padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(0.2))
	model.add(layers.Conv2DTranspose(128,3,2,padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(0.2))
	model.add(layers.Conv2DTranspose(64,3,2,padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(0.2))
	model.add(layers.Conv2DTranspose(32,3,2,padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(0.2))
	model.add(layers.Conv2DTranspose(3,3,1,padding='same',activation='tanh'))
	return model

##Petit tips : faire un .summary() pour vérifier qu'on s'est pas trompé dans les dimensions de sortie
define_generator().summary()

def train_step(real_images,generator,discriminator,loss,g_opt,d_opt):
  batch_size = tf.shape(real_images)[0]
  global LATENT_DIM
  with tf.GradientTape() as disc_tape:

    ###A compléter###
    latent_vector = tf.random.normal(shape=(batch_size,LATENT_DIM))
    fake_images = generator(latent_vector)
    #Ce sont les prédictions du discriminateur
    real_predictions =  discriminator(real_images)
    fake_predictions = discriminator(fake_images)

    #Les labels sont les vrais labels des images du dataset (0 ou 1)
    real_labels = tf.ones(shape=(batch_size,1))
    fake_labels = tf.zeros(shape= (batch_size,1))

    disc_loss_on_real = loss(real_labels,real_predictions)
    disc_loss_on_fake = loss(fake_labels,fake_predictions)
    disc_loss = disc_loss_on_real + disc_loss_on_fake

  disc_grad = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  d_opt.apply_gradients(zip(disc_grad,discriminator.trainable_variables))
  
  with tf.GradientTape() as gen_tape:

    ###A compléter###
    latent_vector = tf.random.normal(shape=(batch_size,LATENT_DIM))
    fake_images = generator(latent_vector)
    fake_predictions =  discriminator(fake_images)

    #Rappel : on veut comparer les images générées à des 1 pour tromper le discriminateur cette fois
    real_labels = tf.ones(shape=(batch_size,1))
    gen_loss = loss(real_labels,fake_predictions)

  gen_grad = gen_tape.gradient(gen_loss,generator.trainable_variables)
  g_opt.apply_gradients(zip(gen_grad,generator.trainable_variables))
  return gen_loss,disc_loss

def train(dataset,generator,discriminator,epochs,fixed_seed = tf.random.normal((25,LATENT_DIM),seed=42)):
  
  ###A compléter###
  loss = keras.losses.BinaryCrossentropy()
  g_opt = keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
  d_opt = keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)

  Lgen_loss = []
  Ldisc_loss = []
  X = []
  j = 0

  for epoch in range(epochs):
    progress_bar = tqdm(dataset)
    ##Vu que c'est un dataset tensorflow, on ne peut itérer directement dessus avec son indice. On va juste prendre à chaque fois batch par batch.
    for _,image_batch in enumerate(progress_bar):
        j += 1
        gen_loss, disc_loss = train_step(image_batch,generator,discriminator,loss,g_opt,d_opt)
        
        X.append(j)
        Lgen_loss.append(gen_loss)
        Ldisc_loss.append(disc_loss)

    clear_output(wait=False)
    generate_and_save_plots(X, Lgen_loss, Ldisc_loss) #Définie après, pour générer les courbes des loss
    summarize_performance(generator,fixed_seed) #Définie après, pour afficher les images générées

def summarize_performance(generator,fixed_seed):
  fake_images = generator.predict(fixed_seed)
  fig = plt.figure(figsize=(12,12))
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.axis('off')
    plt.imshow(fake_images[i]*0.5+0.5)
  plt.show()

def generate_and_save_plots(X, Lgen_loss, Ldisc_loss):
    fig = plt.figure(figsize=(4,4))
    plt.plot(X,Lgen_loss, label = 'gen_loss')
    plt.plot(X,Ldisc_loss, label = 'disc_loss')
    plt.legend()
    plt.show()

gen = define_generator()
disc = define_discriminator()
train(x_train,gen,disc,epochs=100)
# %%
