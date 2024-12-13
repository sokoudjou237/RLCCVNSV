import os

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from calculer_vitesse import calculer_vitesse_from_dict
from Tracking_YOLOv3 import tracking
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """
    Implémentation du filtre passe-bande de butterworth

    @:param signal: signal initial a filtré
            lowcut: fréquence de coupure basse
            highcut: fréquence de coupure haute
            fs: taux d'échantillonnage du signal
            order: ordre du filtre
    @:return: signal filtré

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def get_vehicle_intensity(videopath, vehicle_timestamps_, vehicle_speeds, lowcut=1024, highcut=3000, order=5, display_tkinter=False, root=False, r=False, sp=False, c=False):

    """
        Calcule et affiche l'intensité sonore des véhicules en fonction de leur vitesse.
        Notez que cette méthode suppose que le signal est calibré et que l'énergie est directement liée à l'intensité sonore.

        @:param :
            videopath (str): Chemin vers le fichier vidéo.
            vehicle_timestamps (dict): Dictionnaire contenant les ID des véhicules et les intervalles de temps
                                       où ils sont présents (ex. {'car_1': (start_time, end_time)}
            vehicle_speeds  (dict): Dictionnaire contenant les ID des véhicules et leurs vitesses
                                    (ex. {'car_1': vitesse}
            lowcut (int, optional): Fréquence de coupure basse pour le filtre passe-bande. Par défaut 1000 Hz.
            highcut (int, optional): Fréquence de coupure haute pour le filtre passe-bande. Par défaut 3000 Hz.
            order (int, optional): Ordre du filtre passe-bande. Par défaut 5.

        @:return
            vehicle_intensities (numpy.ndarray): Tableau des intensités sonores des véhicules (dB).

        @:raise:
            ValueError: Si le nombre de véhicules dans vehicle_timestamps et vehicle_speeds n'est pas identique.
    """
    vehicle_timestamps = {key: value for key, value in vehicle_timestamps_.items() if key in vehicle_speeds}
    # Vérifier si le nombre de véhicules dans vehicle_timestamps et vehicle_speeds est identique
    if len(vehicle_timestamps) != len(vehicle_speeds):
        raise ValueError("Le nombre de véhicules dans vehicle_timestamps et vehicle_speeds doit être identique")

    # Convertir la vidéo en audio
    video = VideoFileClip(videopath)
    audio = video.audio
    audio_file = videopath.replace(".mp4", ".wav")
    audio.write_audiofile(audio_file)
    # audio_file = "1.wav"
    # Charger l'audio avec librosa
    y, sr = librosa.load(audio_file, sr=None)

    # Vérifier si le signal audio contient des valeurs non finies
    if not np.all(np.isfinite(y)):
        print("Warning: Audio buffer contains non-finite values. Replacing them with zeros.")
        y = np.nan_to_num(y)

    # Afficher le spectre
    # signal, sr = librosa.load('20211201_194000.wav')
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    # plt.title("Spectre du signal audio")
    # plt.colorbar(format='%+2.0f dB')

    # Appliquer un filtre passe-bande
    lowcut = 7000
    highcut = 8000
    order = 5
    filtered_signal = butter_bandpass_filter(y, lowcut, highcut, sr, order)

    # Préparer le graphe
    # fig, ax = plt.subplots(figsize=(10, 5))
    fig, ax = plt.subplots()
    ax.set_title('Intensité sonore en fonction de la vitesse')
    # Modifier les étiquettes des axes
    ax.set_xlabel('Vitesse (km/h)')
    ax.set_ylabel('Intensité sonore (dB)')


    vehicle_intensities = {}

    for vehicle_id, time_range in vehicle_timestamps.items():
        start_time, end_time = time_range

        # Découper l'audio filtré
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        vehicle_audio = filtered_signal[start_sample:end_sample]
        # vehicle_audio = butter_bandpass_filter(vehicle_audio, lowcut, highcut, sr, order)

        # Normaliser le signal audio
        pression_reference = 2e-5
        # vehicle_audio = vehicle_audio/np.max(abs(vehicle_audio))


        # Obtenir l'enveloppe du signal avec la transformée de Hilbert
        analytic_signal = hilbert(vehicle_audio)
        envelope = np.abs(analytic_signal)

        # Calcul de l'énergie du signal à partir de l'enveloppe
        energy = envelope ** 2


        # Trouver le pic de l'enveloppe
        peak_index = np.argmax(envelope)
        peak_value = envelope[peak_index]

        # Obtenir l'énergie au pic
        peak_energy = vehicle_audio[peak_index]**2
        # peak_energy = y[peak_index]**2

        # Convertir l'énergie en décibels (dB):
        peak_intensity = 10 * np.log10(peak_energy)
        # peak_intensity = librosa.amplitude_to_db(np.abs(peak_energy), ref=np.max(y))
        vehicle_intensities[vehicle_id] = round(peak_intensity, 2)
        # vehicle_intensities[vehicle_id] = round(peak_energy, 2)

        # plt.figure(figsize=(12, 4))
        # plt.plot(y, label="Signal original")
        # plt.plot(vehicle_audio, label="Signal filtré")
        # plt.plot(envelope, label="Enveloppe du signal", linestyle='--')
        # plt.plot(peak_index, peak_value, 'ro', label="Pic de l'enveloppe")
        # plt.xlabel("Temps (échantillons)")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.title("Analyse du signal audio")
        # plt.show()

    # Afficher l'intensité sonore en fonction de la vitesse
    vehicle_intensities_array = np.array(list(vehicle_intensities.values()))
    vehicle_speeds = np.array(list(vehicle_speeds.values()))
    ax.scatter(vehicle_speeds, vehicle_intensities_array)
    print('vehicle_intensities:', vehicle_intensities)
    # vehicle_intensities = np.array(list(vehicle_intensities.values()))

    for i, txt in enumerate(vehicle_timestamps.keys()):
        ax.annotate(txt, (vehicle_speeds[i], vehicle_intensities_array[i]))

    # Modifier les limites des axes
    ax.set_ylim([min(vehicle_intensities_array)-0.5, max(vehicle_intensities_array)+0.5])
    ax.set_xlim([min(vehicle_speeds)-1, max(vehicle_speeds)+1])
    plt.tight_layout()

    if display_tkinter:  # Gestion de l'affichage par Tkinter
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=r, rowspan=sp, column=c, padx=10, pady=10)
    else:  # Affichage simple
        plt.show()
    # Supprimer le fichier audio temporaire
    os.remove(audio_file)



if __name__=="__main__":
    # videopath = 'W=1280_H=720_F=60_V=30_10.mp4'
    videopath = '1.mp4'
    vehicle_timestamps = {'car_1': [0.27, 4.83], 'car_2': [10.6, 14.2], 'car_3': [11, 13]}
    # videopath = 'W=1920_H=1440_F=30_3_V=60.mp4'; vehicle_timestamps = {'car_2': [3.47, 6.11],'car_3': [4.54, 7.61]} ; vehicle_speeds = {'car_2': 48.4, 'car_3': 58.0} # same
    # videopath = 'W=1920_H=1440_F=30_2_V=30.mp4'; vehicle_timestamps ={'car_5': [20.59, 26.79]}
    # videopath = 'W=1920_H=1080_F=60_6_V=30.mp4'; vehicle_timestamps ={'car_4': [9.24, 15.48]}
    # videopath = 'W=1920_H=1080_F=60_4_V=50.mp4'; vehicle_timestamps = {'car_1': [4.25, 7.29]}
    # videopath = 'W=1920_H=1080_F=60_3_V=40.mp4'; vehicle_timestamps = {'car_2': [1.2, 5.89]}
    # videopath = 'W=1920_H=1080_F=60_2_V=30.mp4' ; vehicle_timestamps = {'car_1': [4.65, 9.44] ,'car_3': [6.51, 10.69]} #diff 2 voitures
    # videopath = 'W=1920_H=1080_F=30_2_V=80.mp4'; vehicle_timestamps = {'car_3':[2.5, 6.67], 'car_14':[11.14, 13.45]}
    # videopath = 'W=1920_H=1080_F=30_1_V=20.mp4'; vehicle_timestamps = {'car_5': [4.5, 13.41]}
    # videopath = 'W=1280_H=720_F=100_6_V=40.mp4'; vehicle_timestamps = {'car_1': [1.91, 5.56]}
    # videopath = 'W=1280_H=720_F=100_5_V=50.mp4'; vehicle_timestamps = {'car_1': [3.74, 6.25]}
    # videopath = 'W=1280_H=720_F=100_3_V=30.mp4'; vehicle_timestamps = {'car_1': [3.03, 7.74]}
    # videopath = 'W=1280_H=720_F=60_V=60_11.mp4'; vehicle_timestamps = {'car_1': [6.1, 8.21]}
    # videopath = 'W=1280_H=720_F=60_V=30_10.mp4'; vehicle_timestamps = {'car_1': [8.4, 12.1]}
    # videopath = 'W=1280_H=720_F=60_9_V=50.mp4'; vehicle_timestamps = {'car_1': [2.4, 4.8]}
    # videopath = 'W=1280_H=720_F=60_8_V=30.mp4'; vehicle_timestamps = {'car_1': [31.74, 35.1]}

    # dict_v,vehicle_timestamps = tracking(videopath)
    # vehicle_speeds = calculer_vitesse_from_dict(dict_v, ((138, 321), (317, 327))*3, 6,1/60, 2, 100)
    vehicle_speeds = {'car_1': 48.4, 'car_3': 45}
    # vehicle_speeds = {'car_1': 48.4, 'car_2': 58.0}
    vehicle_intensity = get_vehicle_intensity(videopath, vehicle_timestamps, vehicle_speeds, lowcut=7000, highcut=8000, order=5)
    print(vehicle_intensity )

    # vehicle_speedss = {'car_1': 60, 'car_2': 30, 'car_3': 30, 'car_4': 50, 'car_5': 40, 'car_6': 30, 'car_7': 80,
    #                        'car_8': 20, 'car_9': 40, 'car_10': 50, 'car_11': 30, 'car_12': 60, 'car_13': 30,
    #                        'car_14': 50, 'car_15': 30}
    # # vehicle_intensities = {'car_1': 43.97, 'car_2': 37.37, 'car_3': 36.44, 'car_4': 40.84, 'car_5': 37.08,
    # #                       'car_6': 38.78, 'car_7': 38.96, 'car_8': 41.42, 'car_9': 34.97, 'car_10': 36.88,
    # #                       'car_11': 37.1, 'car_12': 37.79, 'car_13': 36.1, 'car_14': 34.82, 'car_15': 37.49}
    #
    # vehicle_intensities = {'car_1': -51.26, 'car_2': -57.86, 'car_3': -91.2, 'car_4': -62.44, 'car_5': -64.63,
    #                       'car_6': -70.83, 'car_7': -57.55, 'car_8': -53.21, 'car_9': -59.03, 'car_10': -57.18,
    #                       'car_11': -71.49, 'car_12': -57.98, 'car_13': -58.36, 'car_14': -69.52, 'car_15': -60.08}
    #
    # fig, ax = plt.subplots()
    # ax.set_title('Intensité sonore en fonction de la vitesse')
    # # Modifier les étiquettes des axes
    # ax.set_xlabel('Vitesse (km/h)')
    # ax.set_ylabel('Intensité sonore (dB)')
    #
    # # Afficher l'intensité sonore en fonction de la vitesse
    # vehicle_intensities_array = np.array(list(vehicle_intensities.values()))
    # vehicle_speeds = np.array(list(vehicle_speedss.values()))
    # ax.scatter(vehicle_speeds, vehicle_intensities_array)
    #
    # # vehicle_intensities = np.array(list(vehicle_intensities.values()))
    #
    # for i, txt in enumerate(vehicle_speedss.keys()):
    #     ax.annotate(txt, (vehicle_speeds[i], vehicle_intensities_array[i]))
    #
    # # Modifier les limites des axes
    # ax.set_ylim([min(vehicle_intensities_array) - 0.5, max(vehicle_intensities_array) + 0.5])
    # ax.set_xlim([min(vehicle_speeds) - 1, max(vehicle_speeds) + 1])
    # plt.tight_layout()
    # plt.show()
