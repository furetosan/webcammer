import cv2
import sys
import logging as log
import datetime as dt

cascades = [
	{ 'file':'haarcascade_eye.xml','label':'eye' },
	{ 'file':'haarcascade_eye_tree_eyeglasses.xml','label':'eyeglasses' },
	{ 'file':'haarcascade_frontalcatface.xml','label':'frontalcatface' },
	{ 'file':'haarcascade_frontalcatface_extended.xml','label':'frontalcatface_extended' },
	{ 'file':'haarcascade_frontalface_alt.xml','label':'frontalface_alt' },
	{ 'file':'haarcascade_frontalface_alt2.xml','label':'frontalface_alt2' },
	{ 'file':'haarcascade_frontalface_alt_tree.xml','label':'frontalface_alt_tree' },
	{ 'file':'haarcascade_frontalface_default.xml','label':'frontalface_default' },
	{ 'file':'haarcascade_fullbody.xml','label':'fullbody' },
	{ 'file':'haarcascade_lefteye_2splits.xml','label':'lefteye_2splits' },
	{ 'file':'haarcascade_licence_plate_rus_16stages.xml','label':'licence_plate_rus_16stages' },
	{ 'file':'haarcascade_lowerbody.xml','label':'lowerbody' },
	{ 'file':'haarcascade_profileface.xml','label':'profileface' },
	{ 'file':'haarcascade_righteye_2splits.xml','label':'righteye' },
	{ 'file':'haarcascade_russian_plate_number.xml','label':'russian_plate_number' },
	{ 'file':'haarcascade_smile.xml','label':'smile' },
	{ 'file':'haarcascade_upperbody.xml','label':'upperbody' },
	]

# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier(cascPath)
for c in cascades:
    c['cascade'] = cv2.CascadeClassifier(c['file'])
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
for c in cascades:
    c['anterior'] = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30)
    #     # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    # )
    for c in cascades:
        c['detects'] = c['cascade'].detectMultiScale(gray)

    # Draw a rectangle around the faces
    for c in cascades:
        if len(c['detects']) > 0:
            for i in c['detects']:
                x, y, w, h = i
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, c['label'], (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255))
    for c in cascades:
        if c['anterior'] != len(c['detects']):
            c['anterior'] = len(c['detects'])
            log.info(c['label']+" detects: "+str(len(c['detects']))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
