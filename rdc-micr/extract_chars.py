import cv2
import numpy as np
def extract_chars(image, charCnts, minW=5, minH=15):
	charIter = charCnts.__iter__()
	rois = []
	locs = []
	while True:
		try:
			c = next(charIter)
			(cX, cY, cW, cH) = cv2.boundingRect(c)
			roi = None
			if cW >= minW and cH >= minH:
				roi = image[cY:cY + cH, cX:cX + cW]
				rois.append(roi)
				locs.append((cX, cY, cX + cW, cY + cH))
			else:
				parts = [c, next(charIter), next(charIter)]
				(sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
					-np.inf)
				for p in parts:
					(pX, pY, pW, pH) = cv2.boundingRect(p)
					sXA = min(sXA, pX)
					sYA = min(sYA, pY)
					sXB = max(sXB, pX + pW)
					sYB = max(sYB, pY + pH)
				roi = image[sYA:sYB, sXA:sXB]
				rois.append(roi)
				locs.append((sXA, sYA, sXB, sYB))
		except StopIteration:
			break
	return (rois, locs)     