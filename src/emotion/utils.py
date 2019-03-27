import os
import random
from fastai.vision import load_learner,open_image
from src.utils import base64_2_array,create_save_dir,save_image_today



def predict_array(self,img):
    """
    img(np.array)
    """
    filedir = save_image_today(img,'upload/face')
    image = open_image(filedir)
    pred = self.predict(image)
    return pred

def load_model(model_path='/home/qnkhuat/data/emotion_compilation_split_no_affectnet'):
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
        img = base64_2_array(data)
    else:
        raise Exception('Wrong input type')
    filedir = save_image_today(img,'upload/raw')
    image = open_image(filedir)
    return image


