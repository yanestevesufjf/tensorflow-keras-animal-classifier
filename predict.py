import cv2
import tensorflow as tf
import os

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    try:
        IMG_SIZE = 100  # 50 in txt-based
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))# resize image to match model's expected sizing
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)# return the image with shaping that TF wants.
    except Exception as e:
        print(e)
        # print("Error in filepath " + filepath)

pasta = os.getcwd() + "\\testing"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
jpgs = [arq for arq in arquivos if arq.lower().endswith(".jpg")]
MODELNAME = '128x2-1640539891-CNN.model'
model = tf.keras.models.load_model(MODELNAME)

for path in jpgs:
    # path = 'C:/Users/yan.esteves/Documents/Experimentos/IATrends/Samples/DogCats/testing/{}.jpg'.format(test)    
    try:        
        prediction = model.predict([prepare(path)]) # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
        if (prediction):
            try:
                print('')
                # print(path.replace('C:\\Users\\yan.esteves\\Documents\\Experimentos\\IATrends\\Samples\\DogCats\\testing\\', ''),'-',CATEGORIES[int(prediction[0][0])])
            except:
                print('')
    except:
        print('')
        # print(prediction)  # will be a list in a list.
        # print(path.replace('C:\\Users\\yan.esteves\\Documents\\Experimentos\\IATrends\\Samples\\DogCats', ''),'-',CATEGORIES[int(prediction[0][0])])
    
# print(jpgs)
# MODELNAME = '64x3-1640489889-CNN.model'
# model = tf.keras.models.load_model(MODELNAME)

# testing = [
#     # dogs
#     '50', '68', '79', 'cachorro-card-3', 'dog-tv', 'stock-rhodesian',
#     # cats
#     '8220', '8222', '8234', 'cat', 'cat2', 'cat3']

# for test in testing:
#     path = 'C:/Users/yan.esteves/Documents/Experimentos/IATrends/Samples/DogCats/testing/{}.jpg'.format(test)    
#     prediction = model.predict([prepare(path)]) # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
#     if (prediction):
#         # print(prediction)  # will be a list in a list.
#         print(test,'-',CATEGORIES[int(prediction[0][0])])
    