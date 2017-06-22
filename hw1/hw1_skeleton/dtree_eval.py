import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score


def evaluate_performance():

    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    x = np.array(data[:, 1:])
    y = np.array([data[:, 0]]).T
    m, n = x.shape
    trails = 100
    folds = 10
    example_count = m
    remainder = m % folds
    standard_results = []
    one_level_results = []
    three_level_results = []

    for trail in range(0, trails):


        # shuffle data
        idx = np.arange(m)
        np.random.seed(13)
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        for fold in range(0, folds):
            beginning = (fold * (example_count / folds)) + (remainder if (fold > remainder) else fold)
            end = ((fold + 1) * (example_count / folds)) + (remainder if ((fold + 1) > remainder) else (fold + 1)) # the end is non inclusive
            mask = np.zeros(example_count, dtype=bool)
            mask[beginning:end] = True

            x_test = x[mask]
            y_test = y[mask]

            x_train = x[~mask]
            y_train = y[~mask]

            standard_clf = tree.DecisionTreeClassifier()
            standard_clf = standard_clf.fit(x_train, y_train)
            standard_y_predict = standard_clf.predict(x_test)
            standard_fold_accuracy = accuracy_score(y_test, standard_y_predict)
            standard_results.append(standard_fold_accuracy)

            one_level_clf = tree.DecisionTreeClassifier()
            one_level_clf.max_depth = 1
            one_level_clf = one_level_clf.fit(x_train, y_train)
            one_level_y_predict = one_level_clf.predict(x_test)
            one_level_fold_accuracy = accuracy_score(y_test, one_level_y_predict)
            one_level_results.append(one_level_fold_accuracy)

            three_level_clf = tree.DecisionTreeClassifier()
            three_level_clf.max_depth = 3
            three_level_clf = three_level_clf.fit(x_train, y_train)
            three_level_y_predict = three_level_clf.predict(x_test)
            three_level_fold_accuracy = accuracy_score(y_test, three_level_y_predict)
            three_level_results.append(three_level_fold_accuracy)

    # assigning statistics
    meanDecisionTreeAccuracy = np.array(standard_results).mean()
    stddevDecisionTreeAccuracy = np.array(standard_results).std()

    meanDecisionStumpAccuracy = np.array(one_level_results).mean()
    stddevDecisionStumpAccuracy = np.array(one_level_results).std()

    meanDT3Accuracy = np.array(three_level_results).mean()
    stddevDT3Accuracy = np.array(three_level_results).std()

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats

# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print "Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")"
    print "Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")"
    print "3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")"