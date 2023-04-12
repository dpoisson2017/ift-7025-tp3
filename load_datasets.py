from math import ceil

import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché
    iris_file = 'datasets/bezdekIris.data'
    iris_data_records = []
    with open(iris_file) as file:
        for line in file:
            columns = line.rstrip().split(",")
            # Convert labels
            columns[4] = conversion_labels[columns[4]]
            iris_data_records.append(columns)

    # Randomize data order
    random.shuffle(iris_data_records)

    train = []
    train_labels = []
    test = []
    test_labels = []

    training_threshold = int(len(iris_data_records) * train_ratio)
    current_record_count = 0
    for record in iris_data_records:
        data_label = record.pop()
        if current_record_count < training_threshold:
            train.append(record)
            train_labels.append(data_label)
        else:
            test.append(record)
            test_labels.append(data_label)
        current_record_count += 1
    
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
       
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)


def load_iris_dataset_folds(numberFolds):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        numberFolds: le nombre de folds total qui divise le dataset


    Retours:
        Cette fonction doit retourner une liste de list.
        Chaque élément de la liste est en fin une ligne dans le fichier
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché
    iris_file = 'datasets/bezdekIris.data'
    iris_data_records = []
    with open(iris_file) as file:
        for line in file:
            columns = line.rstrip().split(",")
            # Convert labels
            columns[4] = conversion_labels[columns[4]]
            iris_data_records.append(columns)

    # Randomize data order
    random.shuffle(iris_data_records)

    return splitInFolds(iris_data_records, numberFolds)



def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché
    wine_file = 'datasets/binary-winequality-white.csv'
    wine_data_records = []
    with open(wine_file) as file:
        for line in file:
            columns = line.rstrip().split(",")
            wine_data_records.append(columns)

    # Randomize data order
    random.shuffle(wine_data_records)

    train = []
    train_labels = []
    test = []
    test_labels = []
    training_threshold = int(len(wine_data_records) * train_ratio)
    current_record_count = 0
    for record in wine_data_records:
        data_label = int(record.pop())
        if current_record_count < training_threshold:
            train.append(record)
            train_labels.append(data_label)
        else:
            test.append(record)
            test_labels.append(data_label)
        current_record_count += 1

	# La fonction doit retourner 4 structures de données de type Numpy.
    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)

def load_wine_dataset_folds(numberFolds):
    """Cette fonction a pour but de lire le dataset Wine

    Args:
        numberFolds: le nombre de folds total qui divise le dataset


    Retours:
        Cette fonction doit retourner une liste de list.
        Chaque élément de la liste est en fin une ligne dans le fichier
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché
    wine_file = 'datasets/binary-winequality-white.csv'
    wine_data_records = []
    with open(wine_file) as file:
        for line in file:
            columns = line.rstrip().split(",")
            wine_data_records.append(columns)

    # Randomize data order
    random.shuffle(wine_data_records)

    return splitInFolds(wine_data_records, numberFolds)


def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    # Le fichier du dataset est dans le dossier datasets en attaché
    abalone_file = 'datasets/abalone-intervalles.csv'
    abalone_data_records = []
    with open(abalone_file) as file:
        for line in file:
            columns = line.rstrip().split(",")
            abalone_data_records.append(columns)

    # Randomize data order
    random.shuffle(abalone_data_records)

    train = []
    train_labels = []
    test = []
    test_labels = []
    training_threshold = int(len(abalone_data_records) * train_ratio)
    current_record_count = 0
    for record in abalone_data_records:
        data_label = int(float(record.pop()))
        if (record[0] == 'M'):
            record[0] = '1'
        elif (record[0] == 'F'):
            record[0] = '0'
        else:
            record[0] = '0.5'
        if current_record_count < training_threshold:
            train.append(record)
            train_labels.append(data_label)
        else:
            test.append(record)
            test_labels.append(data_label)
        current_record_count += 1


    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)


def load_abalone_dataset_folds(numberFolds):
    """Cette fonction a pour but de lire le dataset Abalone

    Args:
        numberFolds: le nombre de folds total qui divise le dataset


    Retours:
        Cette fonction doit retourner une liste de list.
        Chaque élément de la liste est en fin une ligne dans le fichier
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché
    abalone_file = 'datasets/abalone-intervalles.csv'
    abalone_data_records = []
    with open(abalone_file) as file:
        for line in file:
            columns = line.rstrip().split(",")
            if (columns[0] == 'M'):
                columns[0] = '1'
            elif (columns[0] == 'F'):
                columns[0] = '0'
            else:
                columns[0] = '0.5'
            abalone_data_records.append(columns)

    # Randomize data order
    random.shuffle(abalone_data_records)

    return splitInFolds(abalone_data_records, numberFolds)



def splitInFolds(records, numberFolds):
    numberRecords = len(records)
    average = numberRecords // numberFolds
    remainder = numberRecords % numberFolds
    result = []
    start = 0
    for i in range(numberFolds):
        end = start + average + (i < remainder)
        result.append(records[start:end])
        start = end
    return result

def dataFromFolds(records, totalNumberFolds, foldNumberForTesting):
    train = []
    train_labels = []
    test = []
    test_labels = []

    for foldNumber in range(totalNumberFolds):
        recordsFromFold = records[foldNumber]
        for record in recordsFromFold:
            data_label = int(float(record.pop()))

            if foldNumber != foldNumberForTesting:
                train.append(record)
                train_labels.append(data_label)
            else:
                test.append(record)
                test_labels.append(data_label)

    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)