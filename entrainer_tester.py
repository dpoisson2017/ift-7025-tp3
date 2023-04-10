import load_datasets
#importer d'autres fichiers et classes si vous en avez développés
from BayesNaif import BayesNaif
from sklearn.naive_bayes import GaussianNB
from knn import Knn
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

TRAINING_THRESHOLD = 0.8
NUMBER_NEAREST_NEIGHBORS = 5

datasets = ['iris',
            'wine',
            'abalone']



print("Execution du classificateur Naif Baysien")
for dataset in datasets:
    print("Dataset: {}".format(dataset))

    if dataset == 'iris':
        train, train_label, test, test_label = load_datasets.load_iris_dataset(TRAINING_THRESHOLD)
    elif dataset == 'wine':
        train, train_label, test, test_label = load_datasets.load_wine_dataset(TRAINING_THRESHOLD)
    elif dataset == 'abalone':
        train, train_label, test, test_label = load_datasets.load_abalone_dataset(TRAINING_THRESHOLD)

    print("Creation NaifBayes classifier")
    bayesNaif = BayesNaif()
    print("Training NaifBayes classifier")
    bayesNaif.train(train.astype(float, copy=False), train_label)
    print("Predicting NaifBayes classifier")
    bayesNaif.evaluate(train.astype(float, copy=False), train_label)
    print("Evaluating NaifBayes classifier")
    bayesNaif.evaluate(test.astype(float, copy=False), test_label)

    print("Creation scikit GaussianNB classifier")
    scikit_bayes = GaussianNB()
    print("Training scikit GaussianNB classifier")
    scikit_bayes.fit(train.astype(float), train_label)
    print("Predicting scikit GaussianNB classifier")
    results1 = scikit_bayes.predict(train.astype(float, copy=False))
    print(f"mean accuracy with sklearn GaussianNB: {sklearn.metrics.accuracy_score(train_label, results1)}")
    print("Evaluating scikit GaussianNB classifier")
    results1 = scikit_bayes.predict(test.astype(float, copy=False))
    print(f"mean accuracy with sklearn GaussianNB: {sklearn.metrics.accuracy_score(test_label, results1)}")

print("\n")
print("Execution du classificateur K nearest neighbors (KNN)")
for dataset in datasets:
    print("Dataset: {}".format(dataset))

    if dataset == 'iris':
        train, train_label, test, test_label = load_datasets.load_iris_dataset(TRAINING_THRESHOLD)
    elif dataset == 'wine':
        train, train_label, test, test_label = load_datasets.load_wine_dataset(TRAINING_THRESHOLD)
    elif dataset == 'abalone':
        train, train_label, test, test_label = load_datasets.load_abalone_dataset(TRAINING_THRESHOLD)

    print("Creation KNN classifier")
    knneighbours = Knn(k=NUMBER_NEAREST_NEIGHBORS)
    print("Training KNN classifier")
    knneighbours.train(train.astype(float, copy=False), train_label)
    print("Predicting KNN classifier")
    knneighbours.evaluate(train.astype(float, copy=False), train_label)
    print("Evaluating KNN classifier")
    knneighbours.evaluate(test.astype(float, copy=False), test_label)

    print("Creation scikit KNeighbors classifier")
    scikit_knn = KNeighborsClassifier()
    print("Training scikit KNeighbors classifier")
    scikit_knn.fit(train.astype(float), train_label)
    print("Predicting scikit KNeighbors classifier")
    results1 = scikit_knn.predict(train.astype(float, copy=False))
    print(f"mean accuracy with sklearn KNeighbors: {sklearn.metrics.accuracy_score(train_label, results1)}")
    print("Evaluating scikit KNeighbors classifier")
    results1 = scikit_knn.predict(test.astype(float, copy=False))
    print(f"mean accuracy with sklearn KNeighbors: {sklearn.metrics.accuracy_score(test_label, results1)}")