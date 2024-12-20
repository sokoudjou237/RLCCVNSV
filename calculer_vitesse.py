#Projet radar sonore
#Code fonction calculer_vitesse
#


from math import sqrt
from statistics import median
from plot_histogramme import plot_histogramme_vitesse


def calculer_vitesse(list_coordonnees_objet_tracke, coordonnees_reference, dimension_reference_irl, delta_t, seuil_min = 0, seuil_max = 0, debug=False):
    """
    Reçoit :
    -	une liste des coordonnées (x,y) successives de l’objet tracké,
    -	les coordonnées de deux points définissant une référence d’échelle,
    -	la dimension réelle de cet objet,
    -	le delta_t entre deux images.
    - t0 : indique l'instant initial de la vidéo
    - debug : booléen selon que l'on veut afficher un message avec les informations en plus de retourner la valeur

    Retourne un 3 uplet contenant la vitesse moyenne, minimale et maximale de l'objet tracké

    Affiche un graphique représentant l'évolution de la vitesse du véhicule au cours du temps.
    """

    #Calcul de l'échelle
    (x1_ref,y1_ref),(x_2ref,y2_ref) = coordonnees_reference

    dim_ref_video = sqrt((x_2ref-x1_ref)**2+(y2_ref-y1_ref)**2)

    facteur_echelle = dim_ref_video/(float(dimension_reference_irl))

    #Initialisation de la liste contenant les vitesses instantanées de l'objet tracké
    v_liste = []
    for i in range(1,len(list_coordonnees_objet_tracke)):
        x_i,y_i= list_coordonnees_objet_tracke[i-1]
        x_ii,y_ii= list_coordonnees_objet_tracke[i]
        v_instantanee = round((3.6*sqrt((x_ii-x_i)**2+(y_ii-y_i)**2)/delta_t)/facteur_echelle,1) #facteur d'échelle ^2 pour respecter les dimensions

        if v_instantanee<=seuil_max and v_instantanee>seuil_min:
            v_liste.append(v_instantanee)
    if not v_liste==[]:
        v_med = round(median(v_liste),1)
        v_min = min(v_liste)
        v_max = max(v_liste)
    else:
        return (-1,-1,-1)

    if debug :
        print("L'objet tracké a :\n - une vitesse moyenne de : {} m/s\n - une vitesse minimale de : {} m/s \n - une vitesse maximale de : {} m/s.".format(v_med,v_min,v_max))

    #Inclure ici le code pour l'affichage du graphique
    return (v_med, v_min, v_max)


def calculer_vitesse_from_dict(dict_vitesse,coordonnees_reference, dimension_reference_irl, delta_t, seuil_min = 0, seuil_max = 0):
    # On estime ensuite la vitesse des objets
    dict_objets_speed = {}
    vehicle_speeds = {}
    for object_id in dict_vitesse.keys():
        v_med, v_min, v_max = calculer_vitesse(dict_vitesse[object_id], coordonnees_reference, dimension_reference_irl, delta_t, seuil_min, seuil_max)
        dict_objets_speed[object_id] = [v_med, v_min, v_max]
        vehicle_speeds[object_id] = v_med
        print("Objet {} : vitesse mediane, minimale et maximale : {}, {}, {} km/h".format(object_id, v_med, v_min, v_max))
    print(vehicle_speeds)
    return vehicle_speeds, dict_objets_speed

if __name__== "__main__":
    dict_vitesse = {'car_1': [(1166, 516), (1168, 516), (1168, 517), (1168, 518), (1168, 517), (1167, 517), (1153, 517), (1148, 517), (1145, 517), (1144, 517), (1144, 518), (1143, 517), (1143, 518), (1142, 518), (1139, 519), (1137, 519), (1135, 521), (1134, 521), (1130, 520), (1126, 520), (1125, 520), (1124, 520), (1123, 520), (1122, 521), (1122, 522), (1120, 522), (1118, 523), (1116, 522), (1116, 522), (1114, 522), (1113, 522), (1111, 523), (1105, 522), (1103, 522), (1101, 523), (1099, 524), (1097, 525), (1094, 524), (1092, 525), (1089, 525), (1088, 524), (1087, 524), (1085, 524), (1081, 524), (1079, 524), (1078, 524), (1077, 525), (1075, 524), (1073, 524), (1071, 525), (1069, 525), (1065, 525), (1064, 525), (1062, 525), (1056, 526), (1053, 526), (1051, 527), (1048, 527), (1046, 528), (1044, 528), (1041, 527), (1038, 527), (1036, 527), (1036, 527), (1032, 527), (1029, 527), (1028, 527), (1027, 527), (1024, 528), (1023, 527), (1021, 527), (1019, 527), (1016, 527), (1014, 527), (1012, 527), (1005, 527), (1002, 528), (999, 527), (997, 528), (995, 528), (992, 528), (990, 527), (988, 528), (987, 528), (981, 525), (978, 527), (976, 527), (975, 528), (970, 525), (967, 525), (965, 525), (963, 524), (960, 525), (957, 525), (954, 525), (950, 525), (947, 526), (944, 526), (941, 526), (939, 526), (938, 525), (936, 525), (936, 525), (931, 525), (928, 525), (927, 525), (922, 525), (920, 525), (917, 526), (914, 526), (911, 526), (907, 526), (902, 525), (900, 531), (898, 532), (896, 533), (894, 533), (892, 534), (889, 533), (883, 533), (880, 533), (878, 533), (876, 533), (874, 533), (873, 533), (871, 534), (870, 534), (867, 534), (862, 531), (856, 533), (852, 534), (850, 535), (848, 534), (847, 535), (844, 535), (842, 535), (840, 534), (833, 535), (829, 535), (826, 535), (825, 535), (824, 535), (823, 536), (821, 536), (815, 531), (811, 529), (807, 529), (803, 529), (801, 533), (797, 535), (794, 536), (791, 532), (784, 535), (781, 535), (779, 535), (778, 536), (777, 536), (774, 536), (772, 536), (770, 536), (766, 534), (763, 532), (759, 531), (755, 534), (751, 536), (747, 536), (744, 537), (742, 532), (735, 535), (731, 533), (729, 532), (727, 532), (725, 532), (722, 531), (719, 531), (715, 531), (712, 531), (710, 530), (706, 530), (703, 534), (699, 536), (696, 536), (694, 532), (686, 535), (683, 536), (681, 536), (679, 533), (677, 535), (674, 537), (669, 533), (666, 532), (664, 532), (661, 531), (658, 531), (654, 532), (651, 531), (647, 530), (644, 531), (643, 530), (642, 530), (635, 531), (632, 531), (628, 531), (625, 532), (621, 531), (617, 530), (615, 529), (613, 529), (609, 529), (605, 529), (602, 529), (599, 534), (596, 532), (594, 530), (593, 530), (586, 532), (582, 531), (578, 532), (574, 532), (572, 531), (568, 532), (565, 531), (563, 531), (560, 531), (556, 531), (552, 531), (549, 530), (546, 530), (545, 531), (544, 531), (538, 533), (535, 532), (532, 532), (529, 532), (526, 531), (522, 530), (518, 530), (516, 529), (512, 528), (508, 528), (505, 532), (500, 530), (498, 529), (496, 529), (494, 529), (489, 529), (485, 533), (483, 531), (481, 530), (478, 528), (474, 527), (471, 526), (468, 527), (463, 527), (460, 527), (457, 527), (454, 527), (450, 528), (448, 528), (446, 529), (445, 530), (439, 531), (437, 531), (435, 530), (431, 530), (428, 529), (425, 528), (422, 527), (419, 527), (416, 526), (413, 526), (410, 525), (406, 526), (402, 530), (400, 527), (398, 526), (396, 527), (390, 528), (387, 528), (382, 527), (380, 526), (376, 526), (374, 525), (372, 525), (369, 525), (365, 524), (363, 524), (361, 524), (357, 525), (353, 525), (351, 524), (348, 524), (347, 525), (345, 524), (341, 525), (338, 525), (336, 524), (332, 523), (329, 524), (325, 523), (322, 523), (320, 523), (319, 523), (314, 525), (310, 527), (309, 524), (305, 526), (302, 524), (299, 527), (298, 524), (292, 526), (289, 526), (287, 526), (285, 526), (282, 526), (280, 526), (276, 527), (274, 526), (273, 526), (267, 526), (266, 526), (264, 527), (260, 527), (258, 526), (256, 526), (252, 526), (250, 525), (244, 525), (241, 524), (240, 525), (238, 525), (236, 525), (234, 523), (231, 524), (229, 523), (226, 523), (224, 522), (224, 522), (218, 523), (214, 523), (212, 523), (211, 522), (209, 522), (207, 522), (204, 521), (202, 520), (200, 519), (195, 519), (193, 518), (191, 518), (190, 518), (189, 518), (187, 519), (183, 518), (181, 518), (179, 517), (176, 517), (175, 516), (168, 517), (165, 517), (163, 517), (162, 517), (160, 517), (159, 517), (157, 516), (155, 516), (153, 514), (151, 514), (150, 513), (149, 513), (144, 514), (141, 514), (140, 514), (139, 514), (138, 513), (137, 513), (135, 512), (133, 511), (131, 511), (128, 510), (127, 511), (126, 510), (125, 510), (119, 511), (118, 503), (116, 500), (114, 499), (111, 499), (109, 497), (106, 496), (105, 496), (104, 498), (103, 499), (102, 500), (95, 501), (93, 502), (91, 502), (91, 502), (90, 503), (89, 507), (88, 508), (86, 508), (84, 509), (82, 509), (82, 504), (82, 502), (81, 502), (80, 500), (79, 500), (78, 500), (71, 502), (68, 503), (65, 503), (64, 503), (63, 502), (62, 501), (59, 501), (58, 501), (58, 501), (57, 501), (57, 501), (57, 500), (56, 500), (56, 499), (57, 498), (57, 497), (55, 496), (55, 497), (53, 497), (54, 497), (53, 496), (48, 497), (46, 497), (44, 497), (42, 495), (41, 495), (40, 495), (38, 495), (37, 495), (36, 495), (35, 495), (35, 494), (35, 494), (30, 493), (27, 492), (27, 492), (26, 492), (26, 492), (26, 492), (26, 492)], 'car_2': [(1147, 511), (1146, 510), (1144, 511), (1142, 511), (1139, 511), (1137, 513), (1135, 512), (1135, 512), (1134, 512), (1130, 511), (1127, 511), (1124, 512), (1122, 513), (1120, 512), (1118, 513), (1115, 513), (1113, 514), (1111, 513), (1111, 513), (1105, 513), (1102, 512), (1100, 513), (1097, 513), (1094, 514), (1091, 514), (1090, 514), (1087, 514), (1085, 514), (1084, 514), (1080, 515), (1077, 515), (1075, 515), (1072, 516), (1069, 516), (1067, 516), (1064, 516), (1062, 515), (1061, 515), (1055, 515), (1052, 515), (1050, 515), (1047, 516), (1043, 516), (1040, 516), (1037, 516), (1034, 516), (1031, 517), (1029, 517), (1027, 517), (1026, 517), (1023, 518), (1020, 517), (1017, 517), (1015, 516), (1013, 517), (1011, 517), (1005, 516), (1002, 517), (1000, 518), (996, 518), (993, 519), (989, 520), (986, 520), (982, 520), (979, 520), (977, 520), (975, 520), (972, 520), (968, 521), (965, 522), (962, 522), (957, 522), (954, 523), (951, 522), (948, 523), (945, 524), (941, 523), (938, 524), (936, 525), (932, 525), (930, 524), (928, 525), (926, 525), (922, 526), (919, 525), (916, 525), (914, 525), (907, 526), (904, 526), (900, 527), (896, 527), (892, 527), (890, 527), (884, 526), (881, 527), (880, 527), (878, 527), (876, 527), (873, 527), (869, 527), (866, 526), (864, 527), (858, 527), (854, 527), (850, 527), (846, 527), (841, 527), (838, 528), (833, 527), (831, 527), (828, 527), (826, 527), (822, 527), (820, 527), (817, 528), (815, 528), (808, 527), (804, 533), (802, 534), (797, 531), (794, 529), (791, 528), (784, 532), (781, 534), (779, 535), (776, 536), (771, 531), (768, 534), (761, 534), (756, 535), (755, 536), (751, 536), (746, 535), (743, 535), (742, 535), (735, 535), (731, 535), (729, 536), (725, 535), (721, 536), (719, 536), (711, 536), (707, 536), (704, 537), (701, 536), (697, 536), (694, 536), (693, 537), (686, 536), (682, 536), (679, 536), (675, 536), (672, 536), (669, 536), (668, 537), (661, 537), (655, 537), (652, 536), (648, 536), (644, 535), (643, 536), (635, 536), (631, 536), (628, 537), (624, 536), (622, 536), (620, 537), (620, 537), (610, 537), (605, 536), (600, 536), (597, 536), (595, 536), (588, 535), (584, 535), (580, 536), (578, 537), (574, 536), (573, 537), (571, 537), (562, 538), (557, 538), (554, 537), (550, 537), (547, 537), (544, 537), (538, 536), (534, 536), (531, 536), (527, 537), (525, 536), (521, 536), (519, 536), (512, 536), (508, 535), (505, 537), (502, 536), (499, 536), (497, 536), (488, 535), (483, 536), (481, 536), (479, 536), (475, 536), (472, 535), (467, 529), (461, 534), (458, 535), (454, 535), (450, 536), (446, 536), (439, 530), (436, 529), (432, 528), (429, 527), (427, 527), (424, 527), (422, 532), (421, 535), (414, 535), (410, 531), (408, 529), (404, 528), (401, 528), (398, 528), (389, 528), (385, 528), (383, 528), (379, 528), (376, 527), (374, 527), (372, 527), (365, 527), (361, 528), (357, 528), (355, 528), (351, 528), (349, 527), (347, 528), (340, 527), (337, 527), (334, 527), (331, 527), (329, 526), (327, 527), (324, 527), (317, 526), (313, 526), (309, 526), (306, 526), (304, 526), (300, 525), (297, 526), (292, 525), (289, 525), (286, 525), (283, 525), (281, 525), (277, 525), (275, 525), (273, 524), (267, 523), (264, 524), (262, 525), (259, 525), (255, 524), (252, 524), (248, 524), (243, 522), (240, 522), (238, 521), (236, 522), (232, 521), (230, 520), (227, 520), (225, 520), (219, 520), (216, 520), (214, 520), (212, 521), (208, 521), (206, 520), (203, 519), (200, 519), (199, 518), (192, 517), (189, 516), (186, 516), (184, 517), (183, 517), (181, 516), (179, 516), (177, 515), (176, 515), (167, 515), (161, 515), (160, 514), (158, 514), (157, 514), (158, 514), (156, 514), (154, 514), (144, 514), (140, 514), (137, 514), (135, 514), (133, 513), (130, 513), (129, 512), (127, 512), (126, 511), (119, 512), (116, 513), (114, 513), (112, 512), (110, 512), (108, 511), (106, 510), (104, 509), (104, 509), (95, 510), (91, 510), (90, 509), (86, 509), (84, 509), (82, 509), (81, 509), (79, 509), (77, 509), (77, 509), (69, 509), (66, 509), (64, 510), (63, 510), (63, 505), (64, 503), (64, 502), (64, 501), (59, 500), (55, 500), (53, 501), (52, 501), (48, 501), (45, 501), (44, 502), (44, 500), (42, 500), (41, 501), (40, 500), (38, 500), (37, 500), (36, 500), (34, 499), (34, 499), (33, 499), (31, 499), (30, 498), (29, 497), (28, 497), (27, 497), (27, 496), (26, 495), (23, 495), (22, 494), (21, 494), (19, 494)]}

    co_ref =  ((151,227),(221,227))
    dim_ref_irl = 3
    d_t = 1/100

    seuil_min = 5
    seuil_max = 70

    plot_histogramme_vitesse(calculer_vitesse_from_dict(dict_vitesse,co_ref, dim_ref_irl, d_t,  seuil_min, seuil_max)[1])
