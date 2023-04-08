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
import numpy as np

from classifier import Classifier


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class BayesNaif(Classifier):  # nom de la class à changer

    def __init__(self, **kwargs):
        """
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
        self.training_data: np.ndarray
        self.training_labels: np.ndarray
        self.mean = None
        self.variance = None
        self.training_data = None
        self.training_labels = None
        self.possibleClasses = None

    def train(self, train, train_labels):  # vous pouvez rajouter d'autres attributs au besoin
        """
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
        self.training_data = train
        self.training_labels = train_labels
        self.possibleClasses = list(set(train_labels))
        numberClasses = len(self.possibleClasses)
        numberFeatures = len(train[0])
        numberLabels = len(train_labels)

        # Calculate probabilities for every class
        self.classProbabilities = [0.0] * numberClasses
        for c in range(numberClasses):
            classCount = sum(1 for label in train_labels if label == self.possibleClasses[c])
            self.classProbabilities[c] = float(classCount) / numberLabels

        # Calculate probabitilies of each feature for every class
        classMean, classVariance = self.calculateClassMeanVariance(train, train_labels, self.possibleClasses, numberFeatures)
        self.mean = classMean
        self.variance = classVariance

    def predict(self, data):
        """
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
        numberClasses = len(self.possibleClasses)
        m = len(data)
        likelihood = [0.0] * numberClasses
        for c in range(numberClasses):
            # Calculate class probabilities
            classProbabilities = 1.0
            for feature in range(m):
                classProbabilities *= (1.0 / math.sqrt(2.0 * math.pi * self.variance[c][feature])) * \
                                      math.exp(-(data[feature] - self.mean[c][feature]) ** 2 / (
                                              2.0 * self.variance[c][feature]))

            # Calculate probability a posteriori
            likelihood[c] = classProbabilities * self.classProbabilities[c]

            # Return the class with max posteriori probability
            maxLikelihood = max(likelihood)
            return self.possibleClasses[likelihood.index(maxLikelihood)]

    def calculateClassMeanVariance(self, train_data, training_labels, classes, n_features):
        class_mean = [[0.0] * n_features for _ in range(len(classes))]
        class_variance = [[0.0] * n_features for _ in range(len(classes))]
        for i in range(len(classes)):
            rows = [train_data[j] for j in range(len(train_data)) if training_labels[j] == classes[i]]
            for j in range(n_features):
                mean = self.calculateMean([row[j] for row in rows])
                variance = self.calculateVariance([row[j] for row in rows], mean)
                class_mean[i][j] = mean
                class_variance[i][j] = variance
        return class_mean, class_variance

    def calculateMean(self, rows):
        return sum(rows) / float(len(rows))

    def calculateVariance(self, rows, mean):
        return sum((x - mean) ** 2 for x in rows) / float(len(rows))
