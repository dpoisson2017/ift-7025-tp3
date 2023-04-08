"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

from math import sqrt
import numpy as np

from classifier import Classifier


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn(Classifier): #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.training_data: np.ndarray
		self.training_labels: np.ndarray
		self.k_neighbors = kwargs['k']
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
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
		self.possibleClasses = set(train_labels)

        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		distances:list[tuple(float, str)] = []
		#Could be optimised but first draft
		for index, row in enumerate(self.training_data):
			distance = calculateDistance(x, row)
			label = self.training_labels[index]
			distances.append((distance, label))
		distances.sort()
		nearest_neighbours = {}
		for i in self.possibleClasses:
			nearest_neighbours[i] = 0
		for i in range(self.k_neighbors):
			value = distances[i]
			nearest_neighbours[value[1]] += 1
		
		maxValue = 0
		maxClass = None
		for k, v in nearest_neighbours.items():
			if v > maxValue:
				maxClass = k
				maxValue = v
		return maxClass

def calculateDistance(x:np.ndarray, y:np.ndarray):
	numberColumns = x.size
	if numberColumns != y.size:
		raise RuntimeError('Inegal number of columns')
	return np.linalg.norm(x-y)

