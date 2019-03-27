import os
import datetime
import base64
import numpy as np
import cv2


def encode_payload(payload):
    data = data.decode('utf8')
    data = json.loads(data)
    return data


def base64_2_array(base64_data):
    base64_data = base64.b64decode(base64_data)
    nparr = np.fromstring(base64_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    return img


def create_save_dir(root='upload'):
    # store images with folder name by days
    today = datetime.datetime.today()
    save_dir = os.path.join(root,today.strftime('%Y/%m/%d'))
    os.makedirs(save_dir,exist_ok=True)
    return save_dir

def save_image_today(image,root):
    """
    Save image to the folder by date
    """
    assert root in ['upload/raw','upload/face'],"Root folder are not allowed"
    today = datetime.datetime.today()
    save_dir = create_save_dir(root)
    filename = str(int(today.timestamp()))+'_'+ random.randint(0,1000) +'.jpg'
    filedir = os.path.join(save_dir,filename)
    cv2.imwrite(filedir,image)
    return filedir
