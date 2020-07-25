# import Required pacakages
import cv2
import numpy as np
from matplotlib import  pyplot as plt
import imutils
import pandas as pd
import flask
import werkzeug
import tensorflow as tf
# import keras
import os
from tensorflow import keras
# from keras.models import load_model


app = flask.Flask(__name__)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
imgsize=32


@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    # imagefile = flask.request.files['image']
    # if 'image' not in flask.request.files:
    #     return "error"
    # else:
     # return "Flask Server & Android are Working Successfully"
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    # return str(img.shape)
    threshimage = convert(filename)
    # print(threshimage)

    # print(img.shape)
    # return str(img.shape)
    model = tf.keras.models.load_model(PROJECT_HOME+"/assets/ora.h5")
    df=pd.read_csv(PROJECT_HOME+'/assets/dict_output.csv', sep=',',header=None)
    # result = np.argmax(model.predict(img))
    # print(result)
    # alphabet = next(key for key, value in labels_values.items() if value == result) #reverse key lookup
    # original = db[result][2]
    # return str(threshimage)
    # return str(original)
    # print(str(original))
    
  
    # output=predict(threshimage)
    # return str(output)


    # return imagefile
    # # image=
    
    # return "Flask Server & Android are Working Successfully"

def convert(filename):
    # input images
    imag = cv2.imread(filename,cv2.COLOR_BGR2GRAY)
    
    # Noise filtering
    img = cv2.fastNlMeansDenoising(imag,None,10,7,21)
    # For Clearing the shades in the photo
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    #output the new shadeless image 
    # cv2.imwrite('shadows_out.png', diff_img)
    # normalized image gives higher contrast then normal 
    # cv2.imwrite('shadows_out_norm.png', norm_img)

    # Viewing the Image
    # plt.imshow(norm_img,cmap='gray')

    # load the query image, compute the ratio of the old height
    # to the new height, clone it, and resize it
    # image = cv2.imread('shadows_out_norm.png')
    image=norm_img
    # print(image.shape)
    ratio = image.shape[0] / 300.0
    # print(ratio)
    # s=image.shape[0]*.5
    # s=int(s)
    orig = image.copy()
    image = imutils.resize(image, height =300 )
    # convert the image to grayscale, blur it, and find edges
    # in the image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # finding the edges of the image
    edged = cv2.Canny(image, 30, 200)
    # u=edged.copy()
    # plt.imshow(edged)
    # print(image.shape)
    # cv2.imwrite('resized.jpg',image)

    # plt.imshow(orig, cmap='gray')

    #dilate the images to get the Text block

    kernel=np.ones((7,7),np.uint8)
    u = cv2.dilate(edged,kernel,iterations = 5)
    # plt.imshow(u)
    # cnts , _ = cv2.findContours(u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # for c in cnts:
    # cv2.drawContours(u,[cnts],-1,(0,255,0),3)
    # plt.imshow(u)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = 0

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.10 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

   
    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

   

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # print(s)
    # print(rect[0],rect[2])

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # print(diff)
    # print(rect[1],rect[3])
    # print(rect)

    # print(ratio)

    # multiply the rectangle by the original ratio
    rect *= ratio
    # print(rect)

    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    # return warp.shape
    # print(warp.shape)

    # from skimage import exposure

    # convert the warped image to grayscale and then adjust
    # the intensity of the pixels to have minimum and maximum
    # values of 0 and 255, respectively
    # war = exposure.rescale_intensity(warp, out_range = (0, 255))
    # the pokemon we want to identify will be in the top-right
    # corner of the warped image -- let's crop this region out
    # (h, w) = warp.shape
    # (dX, dY) = (int(w * 0.4), int(h * 0.45))
    # crop = warp[10:dY, w - dX:w - 10]
    # save the cropped image to file
    # cv2.imwrite("cropped.png", warp)

    # plt.imshow(warp,cmap='gray')
    # print(warp.shape)
    # cv2.imwrite("cropp.png", war)

    # m=cv2.imread('cropped.png',0)
    # plt.imshow(m)
    # print(m.max())
    # print(m.shape)
    # print(m.min())

    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    thres = cv2.GaussianBlur(warp,(3,3),0)
    # thre = cv2.GaussianBlur(war,(3,3),0)


    # kernal=np.ones((1,1),np.uint8)
    # thres=cv2.erode(warp,kernel,iterations=1)
    # plt.imshow(thres,cmap='gray')

    # thresh=cv2.bitwise_not(warp)
    thresh=cv2.adaptiveThreshold(thres,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,39)

   

    cons= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cons = imutils.grab_contours(cons)
    # for i,b in enumerate(cons):
    x,y,w,h=cv2.boundingRect(cons)
    cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),1)
    outimg = thresh[y:y+h,x:x+w]
    cv2.imwrite('output.png', outimg)
    status,im=resize(outimg,imgsize)
    if(status != -1):
        img = im.reshape(32,32)
        img = im.reshape(1,32,32,1).astype('float32')
        img /= 255
    return img



def resize(im,size):
    try:
        # im = cv2.imread(path, 0)
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
    except:
        print(sys.exc_info()[0])
        print("Error Image : ")
        return -1,[] #error
# def predict(img):
   
# if __name__ == "__main__":
app.run(host="0.0.0.0", port=5000, debug=True)
