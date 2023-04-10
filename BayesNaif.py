"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas,
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées.
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""
import math
import time
from classifier import Classifier

# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class BayesNaif(Classifier): #nom de la class à changer
    """
    C'est un Initializer.
    Vous pouvez passer d'autre paramètres au besoin,
    c'est à vous d'utiliser vos propres notations
    """
    def __init__(self, **kwargs):
        self.training_data = None
        self.training_labels = None
        self.possibleClasses = None
        self.means = None
        self.variances = None
        self.classProbabilities = None

    """
    C'est la méthode qui va entrainer votre modèle,
    train est une matrice de type Numpy et de taille nxm, avec 
    n : le nombre d'exemple d'entrainement dans le dataset
    m : le mobre d'attribus (le nombre de caractéristiques)

    train_labels : est une matrice numpy de taille nx1

    vous pouvez rajouter d'autres arguments, il suffit juste de
    les expliquer en commentaire

    """
    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        startTime = time.time()
        self.training_data = train
        self.training_labels = train_labels
        self.possibleClasses = list(set(train_labels))
        numberTrainCases = len(train)
        numberClasses = len(self.possibleClasses)
        numberFeatures = len(train[0])
        self.means = [[] for _ in range(numberClasses)]
        self.variances = [[] for _ in range(numberClasses)]
        self.classProbabilities = [0 for _ in range(numberClasses)]
        for numClass in range(numberClasses):
            indices = [i for i in range(numberTrainCases) if train_labels[i] == self.possibleClasses[numClass]]
            self.classProbabilities[numClass] = len(indices) / float(numberTrainCases)
            for feature in range(numberFeatures):
                featureValues = [train[clazz][feature] for clazz in indices]
                mean = sum(featureValues) / float(len(featureValues))
                variance = sum([(f - mean)**2 for f in featureValues]) / float(len(featureValues) - 1)
                self.means[numClass].append(mean)
                self.variances[numClass].append(variance)
        print("Elapsed time: " + str(time.time() - startTime))

    """
    Prédire la classe d'un exemple x donné en entrée
    exemple est de taille 1xm
    """
    def predict(self, data):
        maxProbability = -1
        bestClass = -1
        for clazz in range(len(self.possibleClasses)):
            probability = self.classProbabilities[clazz]
            for feature in range(len(self.means[clazz])):
                mean = self.means[clazz][feature]
                variance = self.variances[clazz][feature]
                x = data[feature]
                probability *= (1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-((x - mean)**2) / (2 * variance))
            if probability > maxProbability:
                maxProbability = probability
                bestClass = self.possibleClasses[clazz]
        return bestClass