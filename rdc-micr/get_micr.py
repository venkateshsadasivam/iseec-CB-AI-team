from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import pytesseract
import pandas as pd
import re

# todo create a seq diagram
def reap_info(image):
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
  gray_image = cv2.imread(image,0)
  (h, w,) = gray_image.shape[:2]
  delta = int(h - (h * 0.2))
  bottom = gray_image[delta:h, 0:w]
  Image.fromarray(bottom).show()
  blackhat = cv2.morphologyEx(bottom, cv2.MORPH_BLACKHAT, rectKernel)
  gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
  gradX = np.absolute(gradX)
  Image.fromarray(gradX).show()
  (minVal, maxVal) = (np.min(gradX), np.max(gradX))
  gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
  gradX = gradX.astype("uint8")
  gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
  thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  Image.fromarray(thresh).show()
  thresh = clear_border(thresh)
  Image.fromarray(thresh).show()
  groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  groupCnts = imutils.grab_contours(groupCnts)
  groupLocs = []
  for (i, c) in enumerate(groupCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print(w,h)
    if w >= 50 and h >= 5:
      groupLocs.append((x, y, w, h))
  groupLocs = sorted(groupLocs, key=lambda x:x[0])
  raw_string = []
  for (gX, gY, gW, gH) in groupLocs:
    group = bottom[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    Image.fromarray(group).show()
    raw_string.append(pytesseract.image_to_string(group,lang='mcr',config='--psm 6').replace(' ',''))
  raw_string = " ".join(raw_string)
  print(raw_string)
  appr1_values = value_by_appr1(bottom,groupLocs)
  output = get_values(raw_string,appr1_values)
  return output

def value_by_appr1(bottom,groupLocs):
  roi_x = [i[0] for i in groupLocs]
  roi_y = [i[1] for i in groupLocs]
  roi_w = [i[2] for i in groupLocs]
  roi_h = [i[3] for i in groupLocs]
  bw = cv2.threshold(bottom, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  # todo update this code to original
  predicted_roi = bw[max(roi_y)-5:max(roi_y)+max(roi_h),min(roi_x)-5:max(roi_x)+max(roi_w)+5]
  Image.fromarray(predicted_roi).show()
  predicted_text = pytesseract.image_to_string(predicted_roi,lang='mcr',config='--psm 6').replace(" ",'')
  print(predicted_text)
  return predicted_text

def flush_string(impure_string):
  pure_string = re.sub('[a-zA-Z]',"",impure_string)
  return pure_string

def get_values(micr_text,appr1_values):
  ap2_flag = False
  try:
    cheque_no = re.findall('c+[0-9]+c',micr_text) 
    if len(cheque_no) != 0:
      cheque_no = cheque_no[0]
      micr_text = micr_text.replace(cheque_no,"")
      print(cheque_no)
    else:
      ap2_flag = True
      cheque_no = re.findall('c+[0-9]+c',appr1_values)
      if len(cheque_no) != 0:
        cheque_no = cheque_no[0]
        appr1_values = appr1_values.replace(cheque_no,"")
        print(cheque_no)
      else:
        raise Exception("Cheque No not found !")
  except Exception :
    raise Exception("Cheque No not found !")
  
    
  try:
    transit_inst_no = re.findall('a+[0-9]+d[0-9]+a',micr_text)
    print(transit_inst_no)
    if len(transit_inst_no) != 0:
      transit_inst_no = transit_inst_no[0]
      micr_text = micr_text.replace(transit_inst_no,"")
      transit_no = re.findall('a+[0-9]+d',transit_inst_no)
      transit_no = transit_no[0]
      institution_no = re.findall('d+[0-9]+a',transit_inst_no)
      institution_no = institution_no[0]
    else:
      ap2_flag = True
      transit_inst_no = re.findall('a+[0-9]+d[0-9]+a',appr1_values)
      print(transit_inst_no)
      if len(transit_inst_no) != 0:
        transit_inst_no = transit_inst_no[0]
        appr1_values = appr1_values.replace(transit_inst_no,"")
        transit_no = re.findall('a+[0-9]+d',transit_inst_no)
        transit_no = transit_no[0]
        institution_no = re.findall('d+[0-9]+a',transit_inst_no)
        institution_no = institution_no[0]
      else:
        raise Exception('Transit number or Institution number extraction failed !')
  except Exception:
    raise Exception('Transit number or Institution number extraction failed !')
  if not ap2_flag:
    try:
      acc_no = re.findall('[0-9]+',micr_text)
      acc_no = "".join(acc_no)
    except Exception:
      Exception('Account Number Not found !')
  else:
    try:
      acc_no = re.findall('a+[0-9]+c',micr_text)
      if len(acc_no)==0:
        acc_no = re.findall('a+[0-9]+c',appr1_values)
      acc_no = "".join(acc_no)
    except Exception:
      Exception('Account Number Not found !')
  print(re.findall('a+[0-9]+c',micr_text))
  return {"cheque_no":flush_string(cheque_no),'transit_no':flush_string(transit_no),'institution_no':flush_string(institution_no),'acc_no':acc_no}






# from glob import glob
# files = glob("/home/administrator/Venkatesh/Projects/concentra/rdc/sample_images/all_samples/*")
# count = 1
# opt = []
# for i in files:
#   print(count)
#   print(i)
#   try:
#     opt.append(str(reap_info(i)))

#   except Exception as E:
#     opt.append(E)
#   count+=1

# optdf = pd.DataFrame(opt,columns=['output'])
# optdf['filename'] = [i.split('/')[-1] for i in files]

# optdf.to_excel('hybrid_approach-v2.xlsx')
print(reap_info('/home/administrator/Venkatesh/Projects/concentra/rdc/sample_images/all_samples/sample-16.png'))


