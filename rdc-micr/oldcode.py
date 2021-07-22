from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import pytesseract
from extract_chars import extract_chars
import pandas as pd
import re


def reap_info(image):
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
  gray_image = cv2.imread(image,0)
  (h, w,) = gray_image.shape[:2]
  delta = int(h - (h * 0.2))
  bottom = gray_image[delta:h, 0:w]
  blackhat = cv2.morphologyEx(bottom, cv2.MORPH_BLACKHAT, rectKernel)
  #Image.fromarray(blackhat).show()
  gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
  gradX = np.absolute(gradX)
  (minVal, maxVal) = (np.min(gradX), np.max(gradX))
  gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
  gradX = gradX.astype("uint8")
  gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
  thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  thresh = clear_border(thresh)
  #Image.fromarray(thresh).show()
  groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  groupCnts = imutils.grab_contours(groupCnts)
  groupLocs = []
  for (i, c) in enumerate(groupCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    if w > 50 and h > 15:
      groupLocs.append((x, y, w, h))
  groupLocs = sorted(groupLocs, key=lambda x:x[0])
  roi_x = [i[0] for i in groupLocs]
  roi_y = [i[1] for i in groupLocs]
  roi_w = [i[2] for i in groupLocs]
  roi_h = [i[3] for i in groupLocs]
  bw = cv2.threshold(bottom, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  # opencv_image = cv2.bilateralFilter(bottom,9,75,75)
  #Image.fromarray(cv2.drawContours(bottom,groupCnts,-1,(255,0,0),5)).show() 
  all_opt = []
  appr3 = []
  appr3_2 = []
  for (gX, gY, gW, gH) in groupLocs:
    group = bottom[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    all_opt.append(pytesseract.image_to_string(group,lang='mcr'))
#char contours
    charKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    char_image = cv2.morphologyEx(group, cv2.MORPH_CLOSE, charKernel)
    charCnts = cv2.findContours(char_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    charCnts = imutils.grab_contours(charCnts)
    charCnts = contours.sort_contours(charCnts, method="left-to-right")[0]
    char_list = []
    for c in charCnts:
      (x, y, w, h) = cv2.boundingRect(c)
     # Image.fromarray(group[y:y+h+2,x:x+w+2]).show()
      char_list.append(pytesseract.image_to_string(group[y:y+h+2,x:x+w+2],lang='mcr',config='--psm 10'))
      # cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    appr3.append("".join(char_list))
#
#    
    clone = np.dstack([group.copy()] * 3)
#     refCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     refCnts = imutils.grab_contours(refCnts)
#     refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    (refROIs, refLocs) = extract_chars(group, charCnts)
    [cv2.rectangle(clone,(i[0],i[1]),(i[2],i[3]),(0, 255, 0),2) for i in refLocs]
    
    # Image.fromarray(clone).show()
    ex_opt = []
    for i,j,k,l in refLocs:
      ex_opt.append(pytesseract.image_to_string(group[j:l+2,i:k+2],lang='mcr',config='--psm 10'))
    appr3_2.append("".join(ex_opt))
#     for c in refCnts:
#       (x, y, w, h) = cv2.boundingRect(c)
# #      print((w, h))
# #      print("#"*50)
#       if w >= 5 and h >= 15:
#         #Image.fromarray(clone[y:y+h+2,x:x+w+2]).show()
#         appr3.append(pytesseract.image_to_string(clone[y:y+h+2,x:x+w+2],lang='mcr',config='--psm 10'))
#         #cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
#       else:
#         #cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        
#         appr3.append(' ')
    #Image.fromarray(clone).show()  
  predicted_roi = bw[max(roi_y)-5:min(roi_y)+max(roi_h),min(roi_x)-5:max(roi_x)+max(roi_w)+5]
  #Image.fromarray(predicted_roi).show()
  opt = pytesseract.image_to_string(predicted_roi,lang='mcr')
  all_opt = [i.replace(" ","") for i in all_opt]
  print("approach 1")
  print(opt)
  print("#"*50)
  print("approach 2")
  print(all_opt)
  print("#"*50)
#  print(appr3)
  # appr3 = "".join(appr3)
  # appr3 = ','.join(appr3.split())
  print("approach 3")
  print(appr3)
  print("appr3.2")
  print(appr3_2)
  return opt,all_opt,appr3,appr3_2

appr1 = []
appr2 = []
appr3 = []
appr3_2 = []
for i in range(1,14):
  if i != 4 and i!=6:
    q,w,e,r = reap_info("/home/administrator/Venkatesh/Projects/concentra/rdc/sample_images/sample"+str(i)+".png")
    print(i)
    appr1.append(q)
    appr2.append(w)
    appr3.append(e)
    appr3_2.append(r)

df = pd.DataFrame(appr1,columns=['approach 1'])
df['approach 2'] = appr2
df['approach 3'] = appr3
df['approach 3.2'] = appr3_2

df.to_excel("comparison.xlsx")
