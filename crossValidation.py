import load_datasets
#importer d'autres fichiers et classes si vous en avez développés
from knn import Knn

NUMBER_FOLDS = 10
K_MIN = 5
K_MAX = 10

datasets = ['iris',
            'wine',
            'abalone']

bestKValueByDataset = {'iris': -1, 'wine': -1, 'abalone': -1}

print("Starting cross-validation to find best k-value")
for dataset in datasets:
    print("Dataset: {}".format(dataset))
    l_errors = list()
    for kValue in range(K_MIN, K_MAX):
        sumScores = 0
        for fold in range(NUMBER_FOLDS):
            print("Calculating for k = " + str(kValue) + " fold = " + str(fold))
            if dataset == 'iris':
                dataFolds = load_datasets.load_iris_dataset_folds(NUMBER_FOLDS)
            elif dataset == 'wine':
                dataFolds = load_datasets.load_wine_dataset_folds(NUMBER_FOLDS)
            elif dataset == 'abalone':
                dataFolds = load_datasets.load_abalone_dataset_folds(NUMBER_FOLDS)
            train, train_label, test, test_label = load_datasets.dataFromFolds(dataFolds, NUMBER_FOLDS, fold)
            knneighbours = Knn(k=kValue)
            knneighbours.train(train.astype(float, copy=False), train_label)
            knneighbours.evaluate(test.astype(float, copy=False), test_label)
            sumScores += knneighbours.f1_score
        avgScore = float(sumScores) / NUMBER_FOLDS
        l_errors.append((avgScore, kValue))

    # Take the kValue of the highest F1 score
    bestK = sorted(l_errors, reverse=True)[0][1]
    print("best K = " + str(bestK))
    bestKValueByDataset[dataset] = bestK

print("best K value calculated by cross-validation:")
print(bestKValueByDataset)