import tensorflow as tf
from keras.models import load_model
import efficientnet.keras as efn
import pandas as pd
import numpy as np
import time

from PIL import Image
import matplotlib.pyplot as plt

def rle2mask(rle, width, target_size=None):
    if target_size == None:
        target_size = width

    rle = np.array(list(map(int, rle.split())))
    label = np.zeros((width*width))
    
    for start, end in zip(rle[::2], rle[1::2]):
        label[start:start+end] = 1
        
    #Convert label to image
    label = Image.fromarray(label.reshape(width, width))
    #Resize label
    label = label.resize((target_size, target_size))
    label = np.array(label).astype(float)
    #rescale label
    label = np.round((label - label.min())/(label.max() - label.min()))
    
    return label.T

def mask2rle(mask, orig_dim=160):
    #Rescale image to original size
    size = int(len(mask.flatten())**.5)
    n = Image.fromarray(mask.reshape((size, size))*255.0)
    n = n.resize((orig_dim, orig_dim))
    n = np.array(n).astype(np.float32)
    #Get pixels to flatten
    pixels = n.T.flatten()
    #Round the pixels using the half of the range of pixel value
    pixels = (pixels-min(pixels) > ((max(pixels)-min(pixels))/2)).astype(int)
    pixels = np.nan_to_num(pixels) #incase of zero-div-error
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)
    
class ImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, file_path, batch_size=32, train=False, size=256):
        self.df = df.reset_index(drop=True)
        self.dim = size
        self.train = train
        self.file_path = file_path
        if self.train: self.batch_size = batch_size // 4
        else: self.batch_size = batch_size
        self.pref = 'train'
    
    def __len__(self):
        return np.ceil(len(self.df) / self.batch_size).astype(int)
    
    def __getitem__(self, idx):
        batch_x = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size].id.values
        
        if not self.train:
            X = np.zeros((batch_x.shape[0], self.dim, self.dim, 3))
            
            for i in range(batch_x.shape[0]):
                image = Image.open(self.file_path)
                image = image.resize((self.dim, self.dim))
                image = np.array(image) / 255.
                X[i,] = image
                
            return X
                
        else:
            batch_y = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size].rle.values
            batch_w = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size].img_width.values
            #print(batch_y, batch_w)
            X = np.zeros((batch_x.shape[0]*4, self.dim, self.dim, 3))
            Y = np.zeros((batch_x.shape[0]*4, self.dim, self.dim, 1))
            
            for i in range(batch_x.shape[0]):
                image = Image.open(self.file_path)
                image = image.resize((self.dim, self.dim))
                image = np.array(image)
                rle = rle2mask(batch_y[i], batch_w[i], self.dim)
                rle = rle.reshape((self.dim, self.dim, 1))

                for n, (h, v) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                    X[i*4 + n, :, :, :], Y[i*4 + n, :, :, :] = self.augumention(image,rle)
                    
            return X, Y

epochs = 4
image_size = 512
batch_size = 4

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

test_df = pd.read_csv('./test.csv')
sub = {'id':[], 'rle':[]}

def predict_organ(organ, model_name, file_path):
    test_X = test_df[test_df.organ == organ].reset_index(drop=True)
    if len(test_X) == 0: return #Skip organs without item in test
        
    model = load_model(f"./{model_name}/{organ}_model.h5", compile=False)
    test_loader = ImageDataGenerator(test_X, file_path, batch_size, False, 512)
    start = time.time()
    preds = model.predict(test_loader)
    end = time.time()
    rle = [mask2rle(m, d) for m,d in zip(preds.round(), test_X.img_width)]
    sub['id'] += test_X.id.values.tolist()
    sub['rle'] += rle

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title('Orginal')
    plt.imshow(test_loader[0][0])
    plt.subplot(1, 2, 2)
    plt.title('Predicted')
    plt.imshow(test_loader[0][0])
    plt.imshow(preds[0], cmap='coolwarm', alpha=0.5)

    file_name = f'static/predicts/{model_name}_{organ}.png'
    plt.savefig(file_name)

    time_taken = end - start
    return time_taken
