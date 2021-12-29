import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
import pickle
import time

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0 #Normaliza

# Número de camadas densas em 0 pode ser interessante para escolha binária (dog&cat)
dense_layers = [0] # 2
layer_sizes = [64]
conv_layers = [2] # 3

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()) )
            # print(NAME)
            model = Sequential()
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:])) #Adiciono uma convulation layer
            # print('input shape')
            # print(X.shape[1:])
            # break;
            # model.add(Conv2D(layer_size, (3,3), input_shape = (100, 100, 1) )) #Adiciono uma convulation layer
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
            
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            #epochs=3, 
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model.fit(X, y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[tensorboard])
            model.save('{}x{}-{}-CNN.model'.format(layer_size, conv_layer, int(time.time())))

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# # more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.datasets import cifar10
# import pickle
# import time

# NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

# #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# X = pickle.load(open("X.pickle", "rb"))
# y = pickle.load(open("y.pickle", "rb"))

# X = X/255.0 #Normaliza

# model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) #Adiciono uma convulation layer
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# for l in range(conv_layer-1)
# model.add(Conv2D(64, (3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(64))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# # 10 Layers

# model.compile(loss='binary_crossentropy',
#         optimizer='adam',
#         metrics=['accuracy'])

# #epochs=3, 
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
# model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3, callbacks=[tensorboard])