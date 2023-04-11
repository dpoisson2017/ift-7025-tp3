import sklearn
def evaluate(results, labels):
    rates = {"TP":0, "FP":0, "TN":0, "FN":0}
    confusion_matrixes = dict()
    for i in set(labels):
        confusion_matrixes[i] = rates.copy()
    for index, resultasArray in enumerate(results):
        result = resultasArray.item()
        real_class = labels[index]
        if result == real_class:
            for i in confusion_matrixes:
                if(i == real_class):
                    confusion_matrixes[real_class]["TP"] += 1
                else:
                    confusion_matrixes[i]["TN"] += 1
        else:
            for i in confusion_matrixes:
                if(i == real_class):
                    confusion_matrixes[real_class]["FN"] += 1
                elif(i == result):
                    confusion_matrixes[result]["FP"] += 1
                else:
                    confusion_matrixes[i]["TN"] += 1
    
    v:dict
    accuracies = []
    for k, v in confusion_matrixes.items():
        print(f"Evaluation metrics for class: {k}")
        accuracy = (v["TP"] + v["TN"])/(sum(v.values()))
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy}")
        try:
            precision = (v["TP"])/(v["TP"] + v["FP"])
        except:
            precision = "No true positive or false positive"
        print(f"Precision: {precision}")
        try:
            recall = (v["TP"])/(v["TP"] + v["FN"])
        except:
            recall = "No true positive or false negative"
        print(f"Recall: {recall}")
        try:
            f1score = 2 * ((precision * recall) / (precision + recall))
        except:
            f1score = "recall or precision was invalid"
        print(f"F1-score: {f1score}\n")
        matrix = f"""Confusion matrix
    Predicted
Positive	|    Negative
|---------------|---------------|
|	{v["TP"]}	|	{v["FN"]}	| Positive
|---------------|---------------|		Actual
|	{v["FP"]}	|	{v["TN"]}	| Negative
|---------------|---------------|\n"""
        print(matrix)
    print(f"mean accuracy: {sklearn.metrics.accuracy_score(results, labels)}")