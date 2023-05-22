import cv2, sys, numpy, os
from flask import Flask, render_template, Response
import RPi.GPIO as GPIO
import time
#Thư viện led
from gpiozero import  LED

#RELAY = 17
 
# Den ngoai nha
ledout = LED(13)

#Den trong nha
ledin = LED(21)
# Cua
servo_pin = 14

# Tat thong bao chan da duoc sư dung roi
GPIO.setwarnings(False)
# Dinh dang cach cam chan GPIO
GPIO.setmode(GPIO.BCM)

#GPIO.setup(RELAY, GPIO.OUT)
#GPIO.output(RELAY, 0)
    
# Dinh dang dieu khien cua
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)  

# Dinh dạng flask
app = Flask(__name__)

#Noi hien thi webcam chay 
@app.route('/')
def index():
    #Video streaming home page
    return render_template('index.html')
    

size = 2 # change this to 4 to speed up processing trade off is the accuracy
# file su dụng can de nhan dien khuon mat
classifier = 'haarcascade_frontalface_default.xml'
# folder dẻ luu tru anh truoc do da chup
image_dir = 'images'

print("Face Recognition Starting ...")
# Create a list of images,labels,dictionary of corresponding names
# Nhung thong tin cua anh 
(images, labels, names, id) = ([], [], {}, 0)

#Chay vong lap truy cap load cac anh co trong folder images
# Get the folders containing the training data
for (subdirs, dirs, files) in os.walk(image_dir):

    # Loop through each folder named after the subject in the photos
    for subdir in dirs:
        names[id] = subdir
        
        subjectpath = os.path.join(image_dir, subdir)

        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):

            # Skip non-image formats
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
            path = subjectpath + '/' + filename
            label = id

            # Add to training data
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(im_width, im_height) = (100, 80)
time.sleep(2.0)
# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
haar_cascade = cv2.CascadeClassifier(classifier)
# Mở webcam
webcam = cv2.VideoCapture(0) #  0 to use webcam

# ham chuyen doi goc 
def set_angle(angle):
    duty = angle / 18 + 2  # Chuyển đổi góc thành duty cycle
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)  # Đợi 1 giây
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

def process():
    while True:
        # Loop until the camera is working
        rval = False
        while(not rval):
            # Put the image from the webcam into 'frame'
            (rval, frame) = webcam.read()
            if(not rval):
                print("Failed to open webcam. Trying again...")
        
        startTime = time.time()
        # Dat thoi gian  
        # Flip the image (optional)
        frame=cv2.flip(frame,1) # 0 = horizontal ,1 = vertical , -1 = both

        # Convert to grayscalel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to speed up detection (optinal, change size above)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces and loop through each one
        faces = haar_cascade.detectMultiScale(mini)
        
        if len(faces) == 0:
            print("Unknown - door lock")
        else:
            for i in range(len(faces)):
                face_i = faces[i]

                # Coordinates of face after scaling back by size
                (x, y, w, h) = [v * size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                start =(x, y)
                end =(x + w, y + h)
                # Try to recognize the face
                prediction = model.predict(face_resize)
                cv2.rectangle(frame,start , end, (0, 255, 0), 3) # creating a bounding box for detected face
                cv2.rectangle(frame, (start[0],start[1]-20), (start[0]+120,start[1]), (0, 255, 255), -3) # creating  rectangle on the upper part of bounding box

                # Neu dinh dang duoc guong mat nho hon 90 thì sẽ mở cua, bat den trong va ngoai nha va se dong cua trong 5 giay
                #for i in prediction[1]
                if prediction[1]<90 :  # note: 0 is the perfect match  the higher the value the lower the accuracy
                    cv2.putText(frame,' %s - unlock - %.0f' % (names[prediction[0]],prediction[1]),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),thickness=2)
                    print('%s - %.0f' % (names[prediction[0]],prediction[1]))
                    time.sleep(1)
                    print("Mo")
                    ledout.on()
                    ledin.on()
                    #GPIO.output(RELAY, 0)
                    set_angle(0)
                    time.sleep(5)
                    ledout.off()
                    ledin.off()
                    #GPIO.output(RELAY, 1)
                    set_angle(90)
                    print("Dong")
                else:
                    # Cho ra thong bao khong nhan dang duoc guong mặt
                    cv2.putText(frame,("Unknown {} ".format(str(int(prediction[1])))),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),thickness=2)
                    print("Unknown -",prediction[1])
        
        endTime = time.time()
        
        fps = 1/(endTime-startTime)   
        cv2.rectangle(frame,(30,48),(130,70),(0,0,0),-1)
        cv2.putText(frame,"Fps : {} ".format(str(int(fps))),(34,65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        # Show the image and check for "q" being pressed
        #cv2.imshow('Recognition System', frame)
        ret, buffer = cv2.imencode('.jpg', frame) #compress and store image to memory buffer
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #concat frame one by one and return frame
    
    pwm.stop()
    webcam.release()
    cv2.destroyAllWindows()
    
@app.route('/video_feed')
def video_feed():
    #Video streaming route
    return Response(process(),mimetype='multipart/x-mixed-replace; boundary=frame')

    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000', debug=False,threaded = True)

