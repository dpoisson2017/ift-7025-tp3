"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Classifier: #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.possibleClasses:set[str] = []
		
		
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
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
        
	def predictArray(self, data:np.ndarray) -> np.ndarray:
		results = np.empty((np.shape(data)[0], 1), dtype=int)
		i:np.ndarray
		for index, row in enumerate(data):
			result = self.predict(row)
			results[index] = result
		return results

	def evaluate(self, evaluation_data: np.ndarray, evaluation_labels: np.ndarray):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		rates = {"TP":0, "FP":0, "TN":0, "FN":0}
		class_recognition = dict()
		prediction_results = self.predictArray(evaluation_data)
		for i in self.possibleClasses:
			class_recognition[i] = rates.copy()
		for index, resultasArray in enumerate(prediction_results):
			result = resultasArray.item()
			real_class = evaluation_labels[index]
			if result == real_class:
				for i in class_recognition:
					if(i == real_class):
						class_recognition[real_class]["TP"] += 1
					else:
						class_recognition[i]["TN"] += 1
			else:
				for i in class_recognition:
					if(i == real_class):
						class_recognition[real_class]["FN"] += 1
					elif(i == result):
						class_recognition[result]["FP"] += 1
					else:
						class_recognition[i]["TN"] += 1
		
		v:dict
		print(class_recognition)
		for k, v in class_recognition.items():
			print(f"Evaluation metrics for {k}\n")
			accuracy = (v["TP"] + v["TN"])/(sum(v.values()))
			print(f"Accuracy: {accuracy}\n")
			precision = (v["TP"])/(v["TP"] + v["FP"])
			print(f"Precision: {precision}\n")
			recall = (v["TP"])/(v["TP"] + v["FN"])
			print(f"Recall: {recall}\n")
			f1score = 2 * ((precision * recall) / (precision + recall))
			print(f"F1-score: {f1score}\n")
			print("Confusion matrix")
			matrix = f""" 
			    Predicted
		    Positive	|   Negative
		|---------------|---------------|
	Positive|	{v["TP"]}	|	{v["FN"]}	|
Actual		|---------------|---------------|
	Negative|	{v["FP"]}	|	{v["TN"]}	|
		|---------------|---------------|"""
			print(matrix)