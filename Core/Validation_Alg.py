from DataUtility.DataUtil import DataExamples
import numpy as np


def k_fold_cross_validation(data_examples:DataExamples, k:int, seed:int=0):
    if k <= 1:
        raise ValueError("Fold should be greater than 1")
    if len(data_examples) < k:
        raise ValueError("Fold can't be greater than the number of examples")
    data_examples.Shuffle(seed=seed)

    fold_size = len(data_examples) // k
    remainder = len(data_examples) % k

    folds = []
    start_index = 0
    for i in range(k):
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end_index = start_index + current_fold_size

        fold_data = DataExamples(
            data_examples.Data[start_index:end_index],
            data_examples.Label[start_index:end_index],
            id=data_examples.Id[start_index:end_index] if data_examples.Id is not None else None
        )
        folds.append(fold_data)
        start_index = end_index

    results = []
    for i in range(k):
        test_set = folds[i]
        train_set_data = []
        train_set_label = []
        train_set_ids = [] if data_examples.Id is not None else None

        # Combina i restanti fold per formare il set di addestramento
        for j, fold in enumerate(folds):
            if j != i:
                train_set_data.append(fold.Data)
                train_set_label.append(fold.Label)
                if data_examples.Id is not None:
                    train_set_ids.append(fold.Id)

        # Concatena i dati dei fold rimanenti
        train_set_data = np.concatenate(train_set_data, axis=0)
        train_set_label = np.concatenate(train_set_label, axis=0)
        if data_examples.Id is not None:
            train_set_ids = np.concatenate(train_set_ids, axis=0)

        train_set = DataExamples(train_set_data, train_set_label, id=train_set_ids)
        results.append((train_set, test_set))

    return results
