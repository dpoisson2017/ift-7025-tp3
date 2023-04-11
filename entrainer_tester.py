import load_datasets
#importer d'autres fichiers et classes si vous en avez développés
from BayesNaif import BayesNaif
from sklearn.naive_bayes import GaussianNB
from knn import Knn
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

TRAINING_THRESHOLD = 0.8
NUMBER_NEAREST_NEIGHBORS = 5
NUMBER_BINS = 10
K_MIN = 5
K_MAX = 10

datasets = ['iris',
            'wine',
            'abalone']

def findBestKValue(dataset):
    if dataset == 'iris':
        dataBins = load_datasets.load_iris_dataset_bins(NUMBER_BINS)
    elif dataset == 'wine':
        dataBins = load_datasets.load_wine_dataset_bins(NUMBER_BINS)
    elif dataset == 'abalone':
        dataBins = load_datasets.load_abalone_dataset_bins(NUMBER_BINS)

    l_errors = list()
    for kValue in range(K_MIN, K_MAX):
        print("Calculating for k = " + str(kValue))
        sumScores = 0
        for bin in range(NUMBER_BINS):
            print("Calculating for bin = " + str(bin))
            train, train_label, test, test_label = load_datasets.dataFromBins(dataBins, NUMBER_BINS, bin)
            knneighbours = Knn(k=kValue)
            knneighbours.train(train.astype(float, copy=False), train_label)
            knneighbours.evaluate(test.astype(float, copy=False), test_label)
            sumScores += knneighbours.f1_score
        avgScore = float(sumScores)/NUMBER_BINS
        l_errors.append((avgScore,kValue))

    # Take the kValue of the lowest score
    return sorted(l_errors)[0][1]

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

    print("Starting cross-validation to find best k-value")
    bestK = findBestKValue(dataset)
    print("Best kValue is: " + str(bestK))


    if dataset == 'iris':
        train, train_label, test, test_label = load_datasets.load_iris_dataset(TRAINING_THRESHOLD)
    elif dataset == 'wine':
        train, train_label, test, test_label = load_datasets.load_wine_dataset(TRAINING_THRESHOLD)
    elif dataset == 'abalone':
        train, train_label, test, test_label = load_datasets.load_abalone_dataset(TRAINING_THRESHOLD)

    print("Creation KNN classifier")
    knneighbours = Knn(k=bestK)
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





