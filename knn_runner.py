import time
from sklearn.neighbors import KNeighborsClassifier
from knn import Knn
import load_datasets
import scikit_evaluator

TRAINING_RATIO = 0.8
bestKValueByDataset = {'iris': 5, 'wine': 5, 'abalone': 5}

DATASETS = ['iris',
            'wine',
            'abalone']


def run(datasets, training_ratio):
    print("Execution du classificateur K nearest neighbors (KNN)")
    for dataset in datasets:
        print(f"Dataset: {dataset}")

        if dataset == 'iris':
            train, train_label, test, test_label = load_datasets.load_iris_dataset(training_ratio)
        elif dataset == 'wine':
            train, train_label, test, test_label = load_datasets.load_wine_dataset(training_ratio)
        elif dataset == 'abalone':
            train, train_label, test, test_label = load_datasets.load_abalone_dataset(training_ratio)

        knneighbours = Knn(k=bestKValueByDataset[dataset])
        knneighbours.train(train.astype(float, copy=False), train_label)
        print(f"Evaluating KNN classifier on train {dataset} \n")
        #knneighbours.evaluate(train.astype(float, copy=False), train_label)
        print(f"Evaluating KNN classifier on test {dataset}\n")
        knneighbours.evaluate(test.astype(float, copy=False), test_label)

        scikit_knn = KNeighborsClassifier()
        scikit_knn.fit(train.astype(float), train_label)
        print(f"Evaluating Sci-kit KNN classifier on train {dataset} \n")
        #start_time = time.time()
        #results1 = scikit_knn.predict(train.astype(float, copy=False))
        #elapsed_time = str(time.time() - start_time)
        #scikit_evaluator.evaluate(results1, train_label)
        #print(f"elapsed_time: {elapsed_time}\n")

        print(f"Evaluating Sci-kit KNN classifier on test {dataset} \n")
        start_time = time.time()
        results1 = scikit_knn.predict(test.astype(float, copy=False))
        elapsed_time = str(time.time() - start_time)
        scikit_evaluator.evaluate(results1, test_label)
        print(f"elapsed_time: {elapsed_time}\n")

if __name__ == "__main__":
    run(DATASETS[2:3], TRAINING_RATIO)