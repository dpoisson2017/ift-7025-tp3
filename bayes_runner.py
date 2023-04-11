import time
import sklearn
from sklearn.naive_bayes import GaussianNB
from BayesNaif import BayesNaif
import load_datasets
import scikit_evaluator

TRAINING_RATIO = 0.8

DATASETS = ['iris',
            'wine',
            'abalone']

def run(datasets, training_ratio):
    print("Execution du classificateur Naif Baysien")
    for dataset in datasets:
        print("Dataset: {}".format(dataset))

        if dataset == 'iris':
            train, train_label, test, test_label = load_datasets.load_iris_dataset(training_ratio)
        elif dataset == 'wine':
            train, train_label, test, test_label = load_datasets.load_wine_dataset(training_ratio)
        elif dataset == 'abalone':
            train, train_label, test, test_label = load_datasets.load_abalone_dataset(training_ratio)

        bayesNaif = BayesNaif()
        print(f"Training NaifBayes classifier on train {dataset}")
        bayesNaif.train(train.astype(float, copy=False), train_label)
        #print(f"Evaluating BayesNaif classifier on train {dataset}")
        #print("Predicting NaifBayes classifier")
        #bayesNaif.evaluate(train.astype(float, copy=False), train_label)
        print(f"Evaluating BayesNaif classifier on test {dataset}")
        bayesNaif.evaluate(test.astype(float, copy=False), test_label)

        scikit_bayes = GaussianNB()
        print(f"Training scikit GaussianNB classifier on train {dataset}")
        #print(f"Evaluating Sci-kit GaussianNB classifier on train {dataset} \n")
        start_time = time.time()
        scikit_bayes.fit(train.astype(float), train_label)
        elapsed_time = str(time.time() - start_time)
        print(f"Elapsed time training: {elapsed_time}\n")
        start_time = time.time()
        #results1 = scikit_bayes.predict(train.astype(float, copy=False))
        elapsed_time = str(time.time() - start_time)
        #scikit_evaluator.evaluate(results1, test_label)
        #print(f"Elapsed time predicting: {elapsed_time}\n")

        print(f"\nEvaluating Sci-kit GaussianNB classifier on test {dataset} \n")
        start_time = time.time()
        results1 = scikit_bayes.predict(test.astype(float, copy=False))
        elapsed_time = str(time.time() - start_time)
        print(f"Elapsed Time predicting: {elapsed_time}\n")
        scikit_evaluator.evaluate(results1, test_label)
        
if __name__ == "__main__":
    run(DATASETS, TRAINING_RATIO)