import os
import cv2 as cv
import face_recognition
from threading import Thread
import time
import numpy
import subprocess

dir = os.getcwd()
folder = r"C:\Users\sride\OneDrive\Desktop\sdvk\py\ImageFaceOrganizer\sample_folder"
count = 1

image_files = {}
known_faces = {}

def timer( func ): #decorator to time the function
    def wrapper(*args,**kwargs):
        start_time = time.time()
        func(*args,**kwargs)
        print( func.__name__ + "\t" +str( time.time() -start_time ) )
    return wrapper

@timer
def load_file(): #loading and saving files and folders
    for filename in os.listdir(folder):
        file_path = os.path.join( folder, filename )
        if os.path.isfile( file_path ):
            ext = filename.split(".")[-1]
            if not ext.lower() in ("jpg","png","jpeg") : continue
            image_files[filename] =  cv.imread( file_path, cv.IMREAD_COLOR) 
        else:
            pass


def organize( image, filename ): #comparing and organizing image
    global count
    # detected_faces = face_recognition.face_encodings(image)
    detected_faces = face_recognition.face_locations( image )

    for t, r, b, l in detected_faces:
        cropped_face = numpy.ascontiguousarray(image[t:b,l:r,:])

        for name in known_faces.keys():
            h, w, c = known_faces[name].shape
            known_encoding   = face_recognition.face_encodings(known_faces[name], ((0,w,h,0),) )[0]
            unknown_encoding = face_recognition.face_encodings( cropped_face )
            if not unknown_encoding : continue 
            
            result = face_recognition.compare_faces([known_encoding], unknown_encoding[0], 0.6)
            
            if True in result:
                cv.imwrite(os.path.join( folder, name, filename), image ) 
                break
        else:
            while True:
                name = "person("+str(count)+")"
                dir_path = os.path.join( folder, name )
                if not os.path.exists( dir_path ) : break 
                count+=1
            
            os.mkdir( dir_path )
            cv.imwrite(os.path.join( dir_path, "rf[]"+filename ), cropped_face ) 
            cv.imwrite(os.path.join( dir_path, filename ), image ) 
            known_faces[name] = cropped_face


@timer
def start():
    for filename, image in image_files.items(): #for every image
        organize( image, filename )
        # thread = Thread( target=organize, args=((image,filename)) )
        # thread.start()
        # thread.join()


load_file()
start()

cv.waitKey(0)
cv.destroyAllWindows()
exit()