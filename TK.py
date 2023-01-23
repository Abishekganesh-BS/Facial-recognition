from tkinter import *
import cv2
import os
import numpy as np
from PIL import Image
import pickle
import datetime






main = Tk()

main.title("Facial Attendance Recognizer")

screen_wi = 600
screen_he = 400

screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
screen_w = (screen_width / 2) - (600/2)
screen_h = (screen_height / 2) - (400/2)
main.geometry(f'{screen_wi}x{screen_he}+{int(screen_w)}+{int(screen_h)}')
main.maxsize(600,400)
main.minsize(600,400)

main.configure(bg = '#1f1f1f')


def name():
   
   
        
        
    top = Toplevel(main)
    screen_wi = 300
    screen_he = 200
    top.configure(bg = '#1f1f1f')
    screen_width = top.winfo_screenwidth()
    screen_height = top.winfo_screenheight()
    screen_w = (screen_width / 2) - (300/2)
    screen_h = (screen_height / 2) - (200/2)
    top.geometry(f'{screen_wi}x{screen_he}+{int(screen_w)}+{int(screen_h)}')
    top.maxsize(300,200)
    top.minsize(300,200)

    top.title("Name")
    
    entername = LabelFrame(top,text=" Enter Your Name ",fg="white",bg="#1f1f1f")
    entername.place(y=10,relx=0.5,anchor=N,width=200,height=75)
    
    entername.configure(font=('Segoe UI',8))
    
    
    
            
                
    
    def check():
        nameID = a
        video = cv2.VideoCapture(0)

        facedetect=cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')

        count=0
          
        os.makedirs(path)
  
                    
        while True:
            ret,frame = video.read()
            faces=facedetect.detectMultiScale(frame,1.3, 5)
            for x,y,w,h in faces:
                count=count+1
                name='./images/'+nameID+'/'+ str(count) + '.jpg'
                print("Creating Images........." +name)
                cv2.imwrite(name, frame)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
                
            cv2.imshow("WindowFrame", frame)
            cv2.waitKey(1)
            
            if count > 500:
                break
        video.release()
        cv2.destroyAllWindows()

        
    
    
    
    def checking():
        global a
        a = nameentry.get()
        global path
        path='images/'+a
        
        isExist = os.path.exists(path)

        
        
            
        if a != '':
            
            if isExist:
                nametaken = Label(top,text="Name Already taken",fg="white",bg="#1f1f1f")
                nametaken.configure(font=('Segoe UI',8))
                nametaken.place(relx=0.5,y=100,anchor = N)
                
            else:
                try:
                    nametaken.destroy()
                    
                except:
                    pass
                top.destroy()
                check()
                
        else:
            pass
            
            
    nameentry = Entry(entername,width=25)
    nameentry.configure(font=('Segoe UI',10))
    
    nameentry.pack(expand = True)
    
    Continue = Button(top, text ="Continue",command=checking)
    Continue.configure(font=('Segoe UI',8))
   
    Continue.place(y=150,relx=0.5,anchor=N,width=100,height=35)
    
def times(time):
    
    x = datetime.datetime.now()
    f = x.strftime("%c")
    time.append(f)
    return time

def recog():

    attendance = open('attendance/attendances.txt','w')
    names = []
    nt=[]
    time = []
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./recognizers/face-trainner.yml")

    labels = {"person_name": 1}
    with open("pickles/face-labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)#camera

    while(True):
   
        ret, frame = cap.read()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
   
            roi_gray = gray[y:y+h, x:x+w] 
            roi_color = frame[y:y+h, x:x+w]

    	
            id_, conf = recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
    	
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    if name not in names:
                        time=times(time)
                       
                        names.append(name)

            img_item = "7.png"
        
            color = (255, 0, 0) 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    	
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    for i in range(len(names)):
        data = [names[i],time[i]]
        nt.append(data)

    for i in nt:
        
        attendance.write(str(i))
        attendance.write('\n')
    attendance.close()
    
def train():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
            for file in files:
                    if file.endswith("png") or file.endswith("jpg"):
                            path = os.path.join(root, file)
                            label = os.path.basename(root).replace(" ", "-").lower()
                            
                            if not label in label_ids:
                                    label_ids[label] = current_id
                                    current_id += 1
                            id_ = label_ids[label]
                            pil_image = Image.open(path).convert("L") 
                            size = (550, 550)
                            final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
                            image_array = np.array(final_image, "uint8")
                            
                            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                            for (x,y,w,h) in faces:
                                    roi = image_array[y:y+h, x:x+w]
                                    x_train.append(roi)
                                    y_labels.append(id_)


    with open("pickles/face-labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("recognizers/face-trainner.yml")
    recog()

    
    
    

head = Label(main,text="Facial Attendance Recognizer",fg="white",bg="#1f1f1f",pady=25)
head.configure(font=('Segoe UI',15))
head.place(relx=0.5,rely=0,anchor = N)

Add = Button(main, text ="Add Person",command=name)
Add.configure(font=('Segoe UI',15))
Add.place(y=125,relx=0.5,anchor=N,width=150,height=50)


Recognize = Button(main, text ="Recognize",command=train)
Recognize.configure(font=('Segoe UI',20))
Recognize.place(y=250,relx=0.5,anchor=N,width=150,height=50)


main.mainloop()











