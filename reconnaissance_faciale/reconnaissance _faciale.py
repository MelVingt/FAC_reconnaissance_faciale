import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
from pathlib import Path
import os
import ntpath


print('[INFO] Démarage Systeme...')
print('[INFO] Import des modèles..')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()

def reconnaisance_visage(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # Enncodage du/des visage(s)
    face_encodings_list, face_locations_list, landmarks_list = vecteurs_visage(rgb_small_frame)
    face_names = []
    top, right, bottom, left = 0, 0, 0,0
    positionZoom = [top, right, bottom, left]
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # Compare le visage de la caméra avec les visages en photo (compare les distances)
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Inconnu"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        #Si le nom est "inconnu", le rectangle sera rouge sans nom
        if name == "Inconnu":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            positionZoom = [top, right, bottom, left]


    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)

    return positionZoom

def vecteurs_visage(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # Détection des visages (On utilise 68 points de détection)
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transformation(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list

def transformation(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


if __name__ == '__main__':
    print('[INFO] Import des visages')
    face_to_encode_path = Path("visage")
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)
    if len(files)==0:
        raise ValueError('No faces detect in the directory: {}'.format(face_to_encode_path))
    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

    known_face_encodings = []
    for file_ in files:
        image = PIL.Image.open(file_)
        image = np.array(image)
        face_encoded = vecteurs_visage(image)[0][0]
        known_face_encodings.append(face_encoded)

    print('[INFO] Démarrage Webcam')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam démarrée')

    positionZoomSave = [0,0,0,0]
    while True:
        ret, frame = video_capture.read()
        #enleve l'effect mirroir
        frame = cv2.flip(frame, 1)
        positionZoom = reconnaisance_visage(frame, known_face_encodings, known_face_names)
        zero = 0
        # AFFICHAGE SANS SUIVI
        cv2.imshow('PRRD - caméra sans suivi ', frame)

        # SUIVI

        frame_zoom = frame
        height, width, channels = frame_zoom.shape
        # plus la variable Zoom est grande moins le zoom sera important
        zoom = 99

        if positionZoom[0] == zero and positionZoom[1] == zero and positionZoom[2] == zero and positionZoom[3] == zero :
            if positionZoomSave[0] == zero and positionZoomSave[1] == zero and positionZoomSave[2] == zero and positionZoomSave[3] == zero :
                cv2.imshow('PRRD - caméra', frame)
            else: #On zomm la ou la dernière fois le prof à été détecté
                minX, maxX = (positionZoomSave[0]), (positionZoomSave[2])
                minY, maxY = (positionZoomSave[3]), (positionZoomSave[1])
                cropped = frame_zoom[int(minX):int(maxX), int(minY):int(maxY)]
                resized_cropped_frame = cv2.resize(cropped, (width, height))
                #cv2.imshow('PRRD - caméra', frame)
                cv2.imshow('PRRD - caméra', resized_cropped_frame)
        else:
            minX, maxX = (positionZoom[0]-zoom*2/3), (positionZoom[2]+zoom*2/3)
            minY, maxY = (positionZoom[3]-zoom), (positionZoom[1]+zoom)

            if minX >= zero and maxX >= zero and minY >= zero and maxY >= zero :
                cropped = frame_zoom[int(minX):int(maxX), int(minY):int(maxY)]
                resized_cropped_frame = cv2.resize(cropped, (width, height))
                cv2.imshow('PRRD - caméra', resized_cropped_frame)

            else : #Gère le zoom quand les postions sont négatives
                minX, maxX = (positionZoom[0]), (positionZoom[2] + zoom * 2 / 3)
                minY, maxY = (positionZoom[3]), (positionZoom[1] + zoom)
                cropped = frame_zoom[int(minX):int(maxX), int(minY):int(maxY)]
                resized_cropped_frame = cv2.resize(cropped, (width, height))
                cv2.imshow('PRRD - caméra', resized_cropped_frame)

            positionZoomSave[0] = minX
            positionZoomSave[1] = maxY
            positionZoomSave[2] = maxX
            positionZoomSave[3] = minY

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stop')
    video_capture.release()
    cv2.destroyAllWindows()
