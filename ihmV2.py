import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import *
import cv2
import os
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from signal_audio import get_vehicle_intensity
from plot_histogramme import *

from moviepy.video.io.VideoFileClip import VideoFileClip
from Tracking_YOLOv3 import tracking
# from Tracking_Faster_RCNN import tracking
from calculer_vitesse import calculer_vitesse_from_dict





def select_file():
    global file_path
    file_path = filedialog.askopenfilename()

    # Récupérer les informations largeur, longueur, fréquence
    video = VideoFileClip(file_path)
    width, height = video.size
    fps = video.fps
    file_label.config(text=f"Vidéo : {os.path.basename(file_path)} ({width}x{height} : {int(round(fps, 0))} FPS)")
    calibrate_button.config(state=tk.NORMAL)  # Activer le bouton "Lancer l'étalonnage"
    start_button.config(state=tk.DISABLED)  # Désactive le bouton Start


root = tk.Tk()
root.title("IHM - Radar vidéo - Projet 4.4")
root.geometry("1920x1080")
root.maxsize(1920, 1080)


# Fonction de validation de saisie
def validate(new_text):
    if new_text == "":
        return True
    try:
        float(new_text)
        return True
    except ValueError:
        print("Merci d'entrer un nombre")
        return False

"""

################################
### AFFICHAGE DES GRAPHIQUES ###
################################

"""

def display_result():
    # On affiche le nouvel état du programme
    status_label.config(text="État : affichage des résultats")
    # Premier graphique en 8 4 2
    plot_boxplot_vitesse(dict_vitesse=dict_tracked, vitesse_limitee=30, display_tkinter=True, root=root, r=8, sp=4, c=2)

    # Deuxième graphique en 8 4 4
    plot_histogramme_vitesse(dict_vitesse=dict_tracked, vitesse_limitee=30, display_tkinter=True, root=root, r=8, sp=4, c=4)

    # Troisième graphique en 8 4 6
    get_vehicle_intensity(file_path, vehicle_timestamps, vehicle_speeds, lowcut=1024, highcut=3000, order=5, display_tkinter=True, root=root, r=8, sp=4, c=6)


def radar():
    # On affiche le nouvel état du programme
    status_label.config(text="État : traitement en cours")
    image_label.pack_forget()
    global dict_tracked, vehicle_speeds, vehicle_timestamps

    dict_v, vehicle_timestamps_ = tracking(file_path)
    vehicle_speeds , dict_tracked = calculer_vitesse_from_dict(dict_v,points_ref,var.get(),1/VideoFileClip(file_path).fps,0, 200)
    vehicle_timestamps = {key: value for key, value in vehicle_timestamps_.items() if key in vehicle_speeds}
    print(vehicle_speeds, vehicle_timestamps)
    print("Hey j'ai lancé le traitement de la vidéo. Distance en pixels : {}, distance réelle : {}m.".format(
        distance_pixels, var.get()))
    time.sleep(2)
    display_result()


def calibrate():
    video = cv2.VideoCapture(file_path)
    success, frame = video.read()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))  # Redimensionner l'image à la moitié de sa dimension réelle

        points = []

        def add_point(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                if len(points) == 2:
                    point_label.config(text=f"Point 1: {points[0]} Point 2: {points[1]}")

                    # calcul de la distance en pixels
                    global distance_pixels
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    global points_ref
                    points_ref = ((x1,y1),(x2,y2))
                    distance_pixels = round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5, 2)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", add_point)
        while len(points) < 2:
            cv2.imshow("Image", frame)
            if cv2.waitKey(20) & 0xFF == 'q':
                break
        cv2.destroyAllWindows()

        print(points_ref)

        # frame = cv2.imread("detect.png")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image)

        image_label.config(image=image)
        image_label.config(width=image.width(), height=image.height())
        image_label.image = image
        image_label.grid(row=2, rowspan=6, column=2, columnspan=5, sticky="nsew", padx=30, pady=30)

        entry.config(state=tk.NORMAL)


# Grid constant

width_divider = 2

root.grid_columnconfigure(0, minsize=300, weight=1)
root.grid_columnconfigure(2, minsize=300, weight=1)
root.grid_columnconfigure(4, minsize=300, weight=1)
root.grid_columnconfigure(6, minsize=300, weight=1)

root.grid_rowconfigure(0, minsize=40, weight=1)
root.grid_rowconfigure(1, minsize=40, weight=1)
root.grid_rowconfigure(2, minsize=40, weight=1)
root.grid_rowconfigure(3, minsize=40, weight=1)
root.grid_rowconfigure(4, minsize=40, weight=1)
root.grid_rowconfigure(5, minsize=40, weight=1)
root.grid_rowconfigure(6, minsize=40, weight=1)
root.grid_rowconfigure(8, minsize=170, weight=4)
root.grid_rowconfigure(9, minsize=40, weight=1)
root.grid_rowconfigure(10, minsize=40, weight=1)

# Grid layout first column (0)

status_label = tk.Label(root, text="Etat : Configuration",  wraplength=250, font=("Helvetica", 25, "underline", "bold"))
status_label.grid(row=0, column=0)

select_file_button = tk.Button(root, text="Sélectionner un fichier vidéo", command=select_file)
select_file_button.grid(row=1, column=0)

calibrate_button = tk.Button(root, text="Lancer l'étalonnage", command=calibrate, state=tk.DISABLED)
calibrate_button.grid(row=2, column=0)

scale_label = tk.Label(root, text="Renseigner l'échelle réelle (en m) :")
scale_label.grid(row=3, column=0)
vcmd = (root.register(validate), '%P')

entry = tk.Entry(root, validate="key", validatecommand=vcmd, state=tk.DISABLED)
entry.grid(row=4, column=0)

start_button = tk.Button(root, text="Lancer le radar", command=radar, state=tk.DISABLED)
start_button.grid(row=5, column=0)

stop_button = tk.Button(root, text="Arrêter et quitter", command=root.quit)
stop_button.grid(row=6, column=0)

# Création d'une ligne de séparation générale
divider = tk.Frame(root, height=2, width=width_divider, bg="black")
divider.grid(row=7, column=0, columnspan=7, sticky="ew")

process_info_label = tk.Label(root, text="Informations : ", font=("Helvetica", 25, "underline", "bold"))
process_info_label.grid(row=8, column=0)

file_label = tk.Label(root, text="", font=("Helvetica", 14), width=30, wraplength=280)
file_label.grid(row=9, column=0)

point_label = tk.Label(root, text="", font=("Helvetica", 14), width=30)
point_label.grid(row=10, column=0)

# Deuxième colonne : Ligne de séparation

# Création d'une ligne verticale de séparation
divider2 = tk.Frame(root, height=2, width=width_divider, bg="black")
divider2.grid(row=0, rowspan=12, column=1, sticky="NS")

divider3 = tk.Frame(root, height=2, width=width_divider, bg="black")
divider3.grid(row=7, rowspan=4, column=3, sticky="NS")

divider4 = tk.Frame(root, height=2, width=width_divider, bg="black")
divider4.grid(row=7, rowspan=4, column=5, sticky="NS")

# Création d'un objet StringVar pour stocker la valeur d'échelle saisie
var = tk.StringVar()
entry.config(textvariable=var)


# Définition de la fonction de trace
def trace(*args):
    if var.get() != "":
        start_button.config(state="normal")
    else:
        start_button.config(state="disabled")


# Surveillance de la valeur de l'input
var.trace("w", trace)

image_label = tk.Label(root)

root.mainloop()
