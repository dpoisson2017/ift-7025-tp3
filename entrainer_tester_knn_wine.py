import numpy as np
import sys
import load_datasets
from knn import Knn # importer la classe du 
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
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
scikit_knn = KNeighborsClassifier()


# Charger/lire les datasets

wine_train, wine_train_label, wine_test, wine_test_label = load_datasets.load_wine_dataset(0.8)


# Entrainez votre classifieur

knneighbours.train(wine_train.astype(float, copy=False), wine_train_label)
print(knneighbours.possibleClasses)
scikit_knn.fit(wine_train.astype(float), wine_train_label)
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


knneighbours.evaluate(wine_train.astype(float, copy=False), wine_train_label)
results1 = scikit_knn.predict(wine_train.astype(float, copy=False))
print(f"mean accuracy with sklearn knn: {sklearn.metrics.accuracy_score(wine_train_label, results1)}")


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
# knneighbours.evaluate(wine_test.astype(float, copy=False), wine_test_label)




