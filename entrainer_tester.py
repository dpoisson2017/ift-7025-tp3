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
TRAINING_RATIO = 0.8
NUMBER_NEAREST_NEIGHBORS = 5

datasets = ['iris',
            'wine',
            'abalone']

bestKValueByDataset = {'iris': 5, 'wine' : 5, 'abalone' : 5}



bayes_runner.run(datasets, TRAINING_RATIO)

print("\n")
knn_runner.run(datasets, TRAINING_RATIO, NUMBER_NEAREST_NEIGHBORS)
