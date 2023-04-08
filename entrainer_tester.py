import numpy as np
import sys
import load_datasets
from knn import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres

k = 5



# Initialisez/instanciez vos classificateurs avec leurs paramètres

knneighbours = Knn(k=k)



# Charger/lire les datasets

iris_train, iris_train_label, iris_test, iris_test_label = load_datasets.load_iris_dataset(0.8)


# Entrainez votre classifieur

knneighbours.train(iris_train.astype(float, copy=False), iris_train_label)
knneighbours.evaluate(iris_train.astype(float, copy=False), iris_train_label)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""




# Tester votre classifieur



"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""






