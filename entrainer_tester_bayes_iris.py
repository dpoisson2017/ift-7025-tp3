import numpy as np
import sys
import load_datasets
from BayesNaif import BayesNaif
#importer d'autres fichiers et classes si vous en avez développés
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics


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


# Initialisez/instanciez vos classificateurs avec leurs paramètres
bayesNaif = BayesNaif()
scikit_bayes = GaussianNB()


# Charger/lire les datasets
iris_train, iris_train_label, iris_test, iris_test_label = load_datasets.load_iris_dataset(0.8)


# Entrainez votre classifieur
bayesNaif.train(iris_train.astype(float, copy=False), iris_train_label)
scikit_bayes.fit(iris_train.astype(float), iris_train_label)


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
bayesNaif.evaluate(iris_train.astype(float, copy=False), iris_train_label)

results1 = scikit_bayes.predict(iris_train.astype(float, copy=False))
print(f"mean accuracy with sklearn GaussianNaiveBayes: {sklearn.metrics.accuracy_score(iris_train_label, results1)}")



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
bayesNaif.evaluate(iris_test.astype(float, copy=False), iris_test_label)
results1 = scikit_bayes.predict(iris_test.astype(float, copy=False))
print(f"mean accuracy with sklearn GaussianNaiveBayes: {sklearn.metrics.accuracy_score(iris_test_label, results1)}")

