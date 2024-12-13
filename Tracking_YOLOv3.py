
from models import *
from utils import utils
import cv2
from sort import *
import time
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from PIL import Image

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416 # taille de l'image (carrés de 416 px)
conf_thres=0.8 # seuil de confiance
nms_thres=0.4 # seuil de suppression non maximal

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def transform_list_to_dict(liste):
    """
    Transforme une liste en dictionnaire

    @:param liste : liste de voitures et possitions

    @:return resultat : dictionnnaire de la liste avec pour keys
                        le numero de la voiture et pour value la
                        liste de toutes ses positions.
    """
    result = {}
    for item in liste:
        car_id = item[0]
        if car_id in result:
            result[car_id].append(item[1])
        else:
            result[car_id] = [item[1]]
    return result

def detect_image(img):
    """
        Détecte des objets dans une image

        @:param img : une image (sous forme d'un objet Image de la bibliothèque PIL)

        @:return detection : contient les coordonnées de tous les objets détectés dans l'image
                            ainsi que leurs labels et scores de confiance
    """
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def tracking(videopath):
    """
        Suivi d’objets dans une séquence vidéo

        @:param videopath : nom de fichier vidéo

        @:return dict_v : dictionnaire contenant les objets détectés et
                          leurs positions au fil des images de la vidéo
                 vehicle_timestamps (dict): Dictionnaire contenant les ID des véhicules et les intervalles de temps
                                            où ils sont présents (ex. {'car_1': (start_time, end_time)}
    """
    vehicle_timestamps = {}  # Ajout d'une liste pour les timestamps
    v = [] # Ajout d'une liste pour les positions des vehicules
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
    vid = cv2.VideoCapture(videopath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    mot_tracker = Sort()

    cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', (700,500))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M','P', '4', 'V')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4','v')
    ret,frame = vid.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,int(fps),(vw,vh))

    frames = 0
    starttime = time.time()
    while(True):
        ret, frame = vid.read()
        if not ret:
            break
        frames += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pilimg = Image.fromarray(frame)

        detections = detect_image(pilimg)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        # Calculez le temps de chaque frame en fonction du nombre de frames et du fps
        current_time = frames / fps

        if detections is not None:

            tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                cls = classes[int(cls_pred)]
                if cls == "car":
                    vehicle_id = cls + "_" + str(int(obj_id))
                    # Vérifiez si l'identifiant du véhicule est déjà présent dans vehicle_timestamps
                    if vehicle_id in vehicle_timestamps:
                        # Mettez à jour l'intervalle de temps
                        vehicle_timestamps[vehicle_id][1] = round(current_time, 2)
                    else:
                        # Ajoutez un nouvel élément au dictionnaire avec l'identifiant et l'intervalle de temps
                        vehicle_timestamps[vehicle_id] = [round(current_time, 2), round(current_time, 2)]

                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    # x = int(x1/unpad_w *img.shape[1] +box_w/2.)
                    # y = int(y1/unpad_h *img.shape[0] +box_w/2.)
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    x = int(x1 + box_w / 2)
                    y = int(y1 + box_h / 2)
                    color = colors[int(obj_id) % len(colors)]

                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    # cv2.putText(frame, ".", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 3) #verifier le centre
                    v.append([cls + "_" + str(int(obj_id)), (x, y)])

        cv2.imshow('Stream', frame)
        outvideo.write(frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 'q':
            break

    totaltime = time.time()-starttime
    # print(frames, "frames", totaltime/frames, "s/frame")

    outvideo.release()
    cv2.destroyAllWindows()
    dict_v = transform_list_to_dict(v)
    #  un dictionnaire filtré qui ne contient que les véhicules ayant un temps supérieur ou égal à 2 secondes
    filtered_vehicle_timestamps = {vehicle_id: timestamps for vehicle_id, timestamps in vehicle_timestamps.items() if
                                   timestamps[1] - timestamps[0] >= 2}

    return dict_v,vehicle_timestamps




if __name__=="__main__":
    videopath = 'W=1280_H=720_F=100_3_V=30.mp4'
    vehicle_timestamps, dictionnaire = tracking(videopath)
    # print(vehicle_timestamps)
