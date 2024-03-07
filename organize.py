import os
import cv2 as cv
import face_recognition

folder = "sample_folder"


image_files = []
references = {}

for filename in os.listdir(folder):
    file_path = os.path.join( os.getcwd(), folder, filename )
    if os.path.isfile( file_path ):
        ext = filename.split(".")[-1]
        if not ext.lower() in ("jpg","png","jpeg") : continue
        image_files.append( cv.imread( file_path, cv.IMREAD_COLOR) )
    else:
        pass


for image in image_files:
    detected_faces = face_recognition.face_encodings(image)
    for face in detected_faces:
        print( face )
#         for known_face in known_faces:
#             known_encoding = face_recognition.face_encodings(known_face)[0]
#             unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

#             results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    #         if match:
    #             add_to_folder
    #         else:
    #             mkdir , copy to folder

    # remove from main folder