# make sure to give name of sample in line 7
import cv2, sys, numpy, os
size = 2
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'
try:
    print("Nhap ten cua ban: ")
    nameFolder = input() # name of person for recognition
    name_class = str(nameFolder)
except:
    print("You must provide a name")
    sys.exit(0)
# tim thu muc da nhap tren trong images nếu không có sẽ tạo thư mục theo tên trên    
path = os.path.join(image_dir, name_class)
if not os.path.isdir(path):
    os.mkdir(path)
#Chiều cao , chiều rộng hình webcam
(im_width, im_height) = (112, 92)
# Opencv thư viện webcam sẽ lấy thông tin file haarcascade_frontalface_default.xml để nhận diện khuôn mặt
haar_cascade = cv2.CascadeClassifier(classifier)
# Mở webcam
webcam = cv2.VideoCapture(0)

# Generate name for image file
# Dặt tên cho ảnh sau khi chụp. Nếu da co anh trong folder thi cộng 1 cho gia tri ten anh
pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1

# Beginning message
print("\n\033[94mThe program will save 20 samples. \
Move your head around to increase while it runs.\033[0m\n")

# The program loops until it has 20 images of the face.
count = 0 # Gia tri dem so lan chup duoc
pause = 0 
count_max = 10
# Max cho so lan chup
while count < count_max:

    # Loop until the camera is working
    # Dat gia tri rval = false
    rval = False
    while(not rval):
        #Lay du lieu anh trong webcam
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    # Get image shape
    #Chieu dai, rong khung hình
    height, width, channels = frame.shape # 640 , 480 ,3

    # thiết kế khung nhận dạng
    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scale down for speed
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Lay guong mat da nhan dang
    # Detect faces
    faces = haar_cascade.detectMultiScale(mini)
    
    # We only consider largest face
    faces = sorted(faces, key=lambda x: x[3])
    # Nếu faces da duoc nhận dạng dung thi sẽ cat guong mat trong khung mini da tao
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Draw rectangle and write name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, name_class, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 255),2)

        # Remove false positives
        # Dieu kien khi khong nhạn dạng duoc gương mat chuan
        if(w * 6 < width or h * 6 < height):
            print("Face too small")
        else:
            # Khi nhận dạng duoc sẽ hien thi thong tin hinh anh thu may duoc chup
            # To create diversity, only save every fith detected image
            if(pause == 0):

                print("Saving training sample "+str(count+1)+"/"+str(count_max))

                # Save image file
                # Luu anh vao file
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                pin += 1
                count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5
    # Ten khung webcam
    cv2.imshow('Sampling', frame)
    # Nhan q de thoat chup anh giua chung 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
