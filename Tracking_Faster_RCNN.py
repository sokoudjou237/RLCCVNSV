
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
# from Model_faster_RCNN import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
import cv2
import torch
from torchvision import transforms
from sort3 import *



img_size = 416

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# download or load the model from disk
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define the torchvision image transforms
transform = transforms.Compose([transforms.ToTensor(), ])


def detect_image(image, model, device, detection_threshold):
    """
           Détecte des objets dans une image

           @:param img : une image (sous forme d'un objet Image de la bibliothèque PIL)

           @:return detection : contient les coordonnées de tous les objets détectés dans l'image
                               ainsi que leurs labels et scores de confiance
    """

    # transform the image to tensor
    detections = []
    image = transform(image).to(device)
    image = image.unsqueeze(0)  # add a batch dimension
    outputs = model(image)  # get the predictions on the image
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    # Converting tensors to array
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()

    # get boxes above the threshold score
    boxes_th = boxes[scores >= detection_threshold].astype(np.int32)
    scores_th = scores[scores >= detection_threshold]

    for y in range(len(boxes_th)):
        # Bboxes, classname & image name
        x1 = boxes_th[y][0]
        y1 = boxes_th[y][1]
        x2 = boxes_th[y][2]
        y2 = boxes_th[y][3]
        # class_num = labels_th[y]
        score = scores_th[y]
        detections.append([x1, y1, x2, y2, score])
        pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    return detections, boxes_th, pred_classes, outputs[0]['labels']


def draw_boxes(boxes, classes, labels, image, obj_id):

    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, classes[i] + "-" + str(int(obj_id)), (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return image


def transform_to_dict(liste):
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


def tracking(videopath):
    """
        Suivi d’objets dans une séquence vidéo

        @:param videopath : nom de fichier vidéo

        @:return dict_v : dictionnaire contenant les objets détectés et
                          leurs positions au fil des images de la vidéo
    """
    vehicle_timestamps = {}  # Ajout d'une liste pour les timestamps
    v = []
    # download or load the model from disk
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(videopath)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    # get the frame width and height
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Stream', (800, 600))
    mot_tracker = Sort()
    # define codec and create VideoWriter object
    out = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"), fourcc, int(fps), (frame_width, frame_height))

    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second
    # load the model onto the computation device
    model = model.eval().to(device)
    # read until end of video
    while (True):
        # capture each frame of the video
        ret, frame = cap.read()
        if not ret:
            break
        # increment frame count
        frame_count += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if ret == True:
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
                detections, boxes, classes, labels = detect_image(frame, model, device, 0.8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculez le temps de chaque frame en fonction du nombre de frames et du fps
            current_time = frame_count / fps

            if detections is not None:
                tracked_objects = mot_tracker.update(detections)

                for x1, y1, x2, y2, obj_id in tracked_objects:

                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
                              (0, 0, 128),
                              (128, 0, 128), (128, 128, 0), (0, 128, 128)]

                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]
                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                    # cv2.putText(frame, "0", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)

                    for i, box in enumerate(boxes):
                        cls = classes[i]
                        if cls == "car":
                            vehicle_id = cls + "_" + str(int(obj_id))
                            # Vérifiez si l'identifiant du véhicule est déjà présent dans vehicle_timestamps
                            if vehicle_id in vehicle_timestamps:
                                # Mettez à jour l'intervalle de temps
                                vehicle_timestamps[vehicle_id][1] = round(current_time, 2)
                            else:
                                # Ajoutez un nouvel élément au dictionnaire avec l'identifiant et l'intervalle de temps
                                vehicle_timestamps[vehicle_id] = [round(current_time, 2), round(current_time, 2)]

                            w = box[2] - box[0]
                            h = box[3] - box[1]
                            x = box[0] + w / 2.
                            y = box[1] + h / 2.
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                            # cv2.rectangle(frame, (x1, y1 - 35), (x1 + 19 + 80, y1), color, -1)
                            cv2.putText(frame, classes[i] + "-" + str(int(obj_id)), (int(box[0]), int(box[1] - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                            # cv2.putText(frame, ".", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                            v.append([classes[i] + "-" + str(int(obj_id)), (x, y)])

                        # print(classes[i] + "-" + str(int(obj_id)))
                    # draw boxes and show current frame on screen
                    # image = draw_boxes(boxes, classes, labels, frame,obj_id)
                # get the end time
                end_time = time.time()
                # get the fps
                # fps = 1 / (end_time - start_time)
                # add fps to total fps
                # total_fps += fps
                # press `q` to exit
                # wait_time = max(1, int(fps / 4))
                # cv2.imshow('Stream', frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    dict_v = transform_to_dict(v)
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    #  un dictionnaire filtré qui ne contient que les véhicules ayant un temps supérieur ou égal à 2 secondes
    filtered_vehicle_timestamps = {vehicle_id: timestamps for vehicle_id, timestamps in vehicle_timestamps.items() if
                                   timestamps[1] - timestamps[0] >= 2}

    return dict_v, vehicle_timestamps


if __name__ == "__main__":
    videopath = '1.mp4'
    vehicle_timestamps, dictionnaire = tracking(videopath)
