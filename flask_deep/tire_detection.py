import cv2
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial import distance as dist

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model("tire_64CNN101.h5")

def _split_into_patches(ar, patch_size, stride):
    h, w, c = ar.shape
    print(h,w,c)
    line_seg = np.zeros((h,w),np.uint8)
    tire_seg = np.zeros((h,w),np.uint8)
    text_seg = np.zeros((h,w),np.uint8)
    # pad img so that it is dividable by the patch size
    pad_h = patch_size - h % patch_size if h % patch_size != 0 else 0
    pad_w = patch_size - w % patch_size if w % patch_size != 0 else 0

    padded = np.pad(ar, [(0, pad_h), (0, pad_w), (0, 0)], mode="reflect")
    padded_h, padded_w, _ = padded.shape
    ppi = ((padded_h - patch_size) // stride + 1) * (
        (padded_w - patch_size) // stride + 1
    )
    patches = np.empty((ppi, patch_size, patch_size, c))
    patch_ix = 0
    for h in range((padded_h - patch_size) // stride + 1):
        for w in range((padded_w - patch_size) // stride + 1):
            patch = padded[
                h * stride : (h * stride) + patch_size,
                w * stride : (w * stride) + patch_size,
            ]
            #img=np.expand_dims(patch, axis=0)
            #img = keras.preprocessing.image.load_img(patch, 
                                            #target_size=(64,64))
            #img_array = keras.preprocessing.image.img_to_array(img)
            # cv2.imshow("patch",patch)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            
            patch = patch[...,::-1].astype(np.float32)
            img_array = tf.expand_dims(patch, 0)  # Create batch axis
          
            predictions = model.predict(img_array)
            ctire = predictions[0]
            ctire = ctire.argmax()

            
    
            if ctire==0:
                pass
                #background
            elif ctire==1: #colorlines
                line_seg[h * stride : (h * stride) + patch_size,w * stride : (w * stride) + patch_size] = 255
            elif ctire==2: #text
                text_seg[h * stride : (h * stride) + patch_size,w * stride : (w * stride) + patch_size] = 255

            if ctire!=0: #tire
                tire_seg[h * stride : (h * stride) + patch_size,w * stride : (w * stride) + patch_size] = 255
            # patches[patch_ix] = patch
            # patch_ix += 1



    return tire_seg, line_seg, text_seg

def center_crop(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = h

    mid_x, mid_y = w//2, h
    offset_x, offset_y = crop_width//2, crop_height
       
    crop_img = img[:, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def get_moments(src):
    cX=0
    cY=0
    src= cv2.copyMakeBorder(src,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))
    dst = src.copy()
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        M = cv2.moments(i)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    print(cX,cY)
    return cX,cY



def main(user_img_path,stride=32,patch_size=64):
    
    line_loc = ""
    text_loc = ""

    user_img_name = user_img_path.split('/')[-1].split('.')[0]
    ori_img = cv2.imread("./flask_deep/static/images/"+user_img_name+".jpg")
    ori_img = center_crop(ori_img, 300)
    ori_y = ori_img.shape[0]
    ori_x = ori_img.shape[1]
    tire_seg, line_seg, text_seg = _split_into_patches(ori_img,patch_size,stride)

    tire_x,tire_y = get_moments(tire_seg)
    line_x,line_y = get_moments(line_seg)
    text_x,text_y = get_moments(text_seg)
    color_patch = ori_img[line_y-20:line_y+20][:][:]
    cv2.imwrite("./flask_deep/static/images/color_patch_"+user_img_name+".jpg",color_patch)

    ori_img = cv2.line(ori_img,(0,tire_y),(ori_x,tire_y),(255,0,0),2)
    ori_img = cv2.line(ori_img,(0,line_y),(ori_x,line_y),(0,255,0),2)
    ori_img = cv2.line(ori_img,(0,text_y),(ori_x,text_y),(0,0,255),2)
    cv2.putText(ori_img,"Center line",(0,tire_y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(ori_img,"Color line",(0,line_y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(ori_img,"Text line",(0,text_y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

    if tire_y<line_y:
        line_loc = "하"
    else:
        line_loc = "상"

    if tire_y<text_y:
        text_loc = "하"
    else:
        text_loc = "상"
    
    f = open("./flask_deep/static/images/"+user_img_name+".txt", 'w')
    data = "color line loc : " + line_loc +"\n"+"text line loc : " + text_loc
    f.write(data)
    f.close()


    cv2.imwrite("./flask_deep/static/images/ori_"+user_img_name+".jpg",ori_img)
    cv2.imwrite("./flask_deep/static/images/tire_seg_"+user_img_name+".jpg",tire_seg)
    cv2.imwrite("./flask_deep/static/images/line_seg_"+user_img_name+".jpg",line_seg)
    cv2.imwrite("./flask_deep/static/images/text_seg_"+user_img_name+".jpg",text_seg)
    print("./flask_deep/static/images/ori_"+user_img_name+".jpg")
    return "./flask_deep/static/images/ori_"+user_img_name+".jpg"