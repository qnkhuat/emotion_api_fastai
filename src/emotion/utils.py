import os
import random
from fastai.vision import load_learner,open_image
from src.utils import bytes2array,create_save_dir,save_image_today
from fastai.vision import Image,pil2tensor
import cv2
import numpy as np

def array2tensor(x):
    """ Return an tensor image from cv2 array """
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    return Image(pil2tensor(x,np.float32).div_(255))

def predict_array(self,img):
    """
    img(np.array)
    """
    #filedir = save_image_today(img,'upload/face')
    #image = open_image(filedir)
    x = array2tensor(img)
    pred = self.predict(x)
    score = {cl:prb for cl,prb in zip(self.data.classes,pred[2].tolist())}
    return {'emotion':str(pred[0]),
            'score':score}

def load_model(model_path='src/emotion/models'):
    """ Load fastai learner but added a new method that can predict with an image """
    import types
    learn = load_learner(model_path)
    # insert a new function to learner
    learn.predict_array = types.MethodType( predict_array, learn )
    return learn

def prepare_input(data,type='base64'):
    """
    Input a base 64 data
    an instance of fastai iamge

    Args : 
        data (dict) : the data from request has been jsonified
    """
    if type=='base64':
        img = bytes2array(data)
    else:
        raise Exception('Wrong input type')
    filedir = save_image_today(img,'upload/raw')
    image = open_image(filedir)
    return image


