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

abalone_train, abalone_train_label, abalone_test, abalone_test_label = load_datasets.load_abalone_dataset(0.8)


# Entrainez votre classifieur

knneighbours.train(abalone_train.astype(float, copy=False), abalone_train_label)
scikit_knn.fit(abalone_train.astype(float), abalone_train_label)
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
knneighbours.evaluate(abalone_train.astype(float, copy=False), abalone_train_label)
results = scikit_knn.predict(abalone_train.astype(float, copy=False))
print(f"mean accuracy with sklearn knn: {sklearn.metrics.accuracy_score(abalone_train_label, results)}")



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

knneighbours.evaluate(abalone_test.astype(float, copy=False), abalone_test_label)





