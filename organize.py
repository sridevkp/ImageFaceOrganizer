import os
import cv2 as cv
import face_recognition
from threading import Thread
import time

dir = os.getcwd()
folder = r"C:\Users\sride\OneDrive\Desktop\sdvk\py\ImageFaceOrganizer\sample_folder"
count = 1

image_files = {}
known_faces = {}

#loading and saving files and folders
for filename in os.listdir(folder):
    file_path = os.path.join( folder, filename )
    if os.path.isfile( file_path ):
        ext = filename.split(".")[-1]
        if not ext.lower() in ("jpg","png","jpeg") : continue
        image_files[filename] =  cv.imread( file_path, cv.IMREAD_COLOR) 
    else:
        pass

#comparing and organizing image
def organize( image, filename ):
    global count
    detected_faces = face_recognition.face_encodings(image)
    for face in detected_faces:
        for name in known_faces.keys():
            known_encoding = face_recognition.face_encodings(known_faces[name])
            result = face_recognition.compare_faces([known_encoding], face)
            if result[0] == True:
                cv.imwrite(os.path.join( folder, name, filename), image ) 
                break
        else:
            name = "person("+str(count)+")"
            dir_path = os.path.join( folder, name )
            while os.path.exists( dir_path ):
                count+=1
                name = "person("+str(count)+")"
                dir_path = os.path.join( folder, name )
            
            print(os.path.join( dir_path, filename ))
            os.mkdir( dir_path )
            cv.imwrite(os.path.join( dir_path, filename ), image ) 
            known_faces[name] = image

#for every image
start_time = time.time()
for filename, image in image_files.items():
    organize( image, filename )
    # thread = Thread( target=organize, args=((image,)) )
    # thread.start()
    # thread.join()
        
time_taken = time.time()-start_time
print("Finished in "+str(time_taken)+" sec")


cv.waitKey(0)