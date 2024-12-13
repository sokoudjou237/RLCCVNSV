#from calculer_vitesse import calculer_vitesse
import matplotlib.pyplot as plt
from random import *
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def create_dict_vitesse(dict_data):
	dict_vitesse = {}

	#On génère un dictionnaire avec les vitesses moyennes

	for car_id in dict_data.values:
		#On calcule les vitesses min, moyenne et max à l'aide de la fonction calculer_vitesse puis on l'ajoute au dictionnaire
		#dict_vitesse[car_id] = calculer_vitesse(dict_data[car_id])
		None

	return dict_vitesse

def generate_fake_dict_vitesse(n=10):
	dict_vitesse = {}
	for i in range(n):
		v_min = randint(20,50)
		v_moy = v_min + randint(0,10)
		v_max = v_moy + randint(0,10)

		dict_vitesse[i]= [v_min, v_moy, v_max]
	#print(dict_vitesse)

	return dict_vitesse

def create_list_vitesse(dict_vitesse):
	"""
	Retourne les 3 listes avec les vitesses min, moyennes et maximales de tous les véhicules trackés
	:param dict_vitesse:
	:return:
	"""

	l_vitesse_min = []
	l_vitesse_moy = []
	l_vitesse_max = []

	for car_id in dict_vitesse:
		l_vitesse_min.append(dict_vitesse[car_id][0])
		l_vitesse_moy.append(dict_vitesse[car_id][1])
		l_vitesse_max.append(dict_vitesse[car_id][2])

	return l_vitesse_min, l_vitesse_moy, l_vitesse_max


def plot_histogramme_vitesse(dict_vitesse=generate_fake_dict_vitesse(30), vitesse_limitee = 30, display_tkinter=False, root=False, r=False, sp=False, c=False):

	"""
	Génère un histogramme représentant le nombre de voitures par plage de vitesse.
	La vitesse limitée (par défaut zone 30) est également affichée.
	:param dict_vitesse:
	:param vitesse_limitee:
	:return:
	"""
	#On génère les 3 listes avec les vitesses min, moyennes et maximales de tous les véhicules trackés

	l_vitesse_min, l_vitesse_moy, l_vitesse_max = create_list_vitesse(dict_vitesse)

	#Création du plot
	fig = plt.figure()

	vmin = min(l_vitesse_min)
	vmax = max(l_vitesse_max)
	bins = np.arange(vmin, vmax+1, 5)
	plt.hist([l_vitesse_min, l_vitesse_moy, l_vitesse_max], bins = bins, color = ['green', 'yellow', 'orange'],
	            edgecolor = 'black', label = ['vitesse min', 'vitesse moy', 'vitesse max'],)
	if vitesse_limitee !=0 :
		plt.axvline(vitesse_limitee, color='red', linestyle='--', label='limite de vitesse : {} km/h'.format(vitesse_limitee))
	plt.ylabel('Nombre de voitures')
	plt.xlabel('Vitesse en km/h')
	plt.title('Nombre de voitures par plage de vitesse')
	plt.legend()

	if display_tkinter:  # Gestion de l'affichage par Tkinter
		canvas = FigureCanvasTkAgg(fig, master=root)
		canvas.draw()
		canvas.get_tk_widget().grid(row=r, rowspan=sp, column=c, padx=10, pady=10)
	else:  # Affichage simple
		plt.show()


def plot_boxplot_vitesse(dict_vitesse, vitesse_limitee = 30, display_tkinter=False, root=False, r=False, sp=False, c=False):
	data1, data2, data3 = create_list_vitesse(dict_vitesse)

	n_cars = len(data1) #nombre de véhicules

	fig, ax = plt.subplots()

	for i, data in enumerate([data1]):
		bp = ax.boxplot(data, positions=[i + 1], vert=False, showfliers=False, showmeans=True)
		for box in bp["boxes"]:
			x, y = box.get_xydata()[0]
			#Calcul des quartiles
			median = np.median(data)
			q1, q3 = np.percentile(data, [25, 75])

			#Calcul des coordonnées
			x_median = bp["medians"][0].get_xdata()[0] - 3.5
			y_median = bp["medians"][0].get_ydata()[0] + 0.25
			ax.annotate("Médiane : {:.2f}".format(median), (x_median, y_median))
			ax.annotate("Q1 : {:.2f}".format(q1), (x_median - 6.25, y - 0.15))
			ax.annotate("Q3 : {:.2f}".format(q3), (x_median + 6.25, y - 0.15))
	if vitesse_limitee != 0:
		plt.axvline(vitesse_limitee, color='red', linestyle='--',
					label='limite de vitesse : {} km/h'.format(vitesse_limitee))
	ax.set_yticklabels(['med'])
	plt.xlabel('Vitesse en km/h')
	plt.title('Répartition des vitesses des {} voitures'.format(n_cars))

	if display_tkinter:  # Gestion de l'affichage par Tkinter
		canvas = FigureCanvasTkAgg(fig, master=root)
		canvas.draw()
		canvas.get_tk_widget().grid(row=r, rowspan=sp, column=c, padx=10, pady=10)
	else:  # Affichage simple
		plt.show()




#plot_histogramme_vitesse(generate_fake_dict_vitesse(30))
#plot_boxplot_vitesse(generate_fake_dict_vitesse(30))