import time
import load_datasets
#importer d'autres fichiers et classes si vous en avez développés
from BayesNaif import BayesNaif
from sklearn.naive_bayes import GaussianNB
from knn import Knn
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import knn_runner
import bayes_runner
training_ratio = 0.8
NUMBER_NEAREST_NEIGHBORS = 5

datasets = ['iris',
            'wine',
            'abalone']




bayes_runner.run(training_ratio, datasets)

print("\n")
knn_runner.run(datasets, training_ratio, NUMBER_NEAREST_NEIGHBORS)