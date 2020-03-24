import numpy as np

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _gini(self, n, m, num_classes):
        return 1.0 - sum(pow((n/m),2) for n in range(num_classes))

    def _best_split(self, X, y):
        '''
        Find best split for a node, meaning average impurity of its two children
        is minimized and less than impurity of current node.
        Return:
            best_i: Index of feature for best split
            best_thr: Best threshold to use for split
        '''

        m = y.size
        if m <= 1:
            return None, None
        
        num_parent = [np.sum(y == c) for c in range(self.num_classes)]

        # Calculate current gini
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_i, best_thr = None, None

        # Loop through features
        for i in range(self.num_features):
            thresholds, classes = zip(*sorted(zip(X[:, i], y)))

            num_left = [0] * self.num_classes
            num_right = num_parent.copy()

            # Loop through possible places to split data
            for j in range(1, m):
                c = classes[j - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / j) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / j) ** 2 for x in range(self.num_classes))
                gini = (j * gini_left + (m - j) * gini_right) / m
                if thresholds[j] == thresholds[j - 1]:
                    continue

                # Set best gini
                if gini < best_gini:
                    best_gini = gini
                    best_i = i

                    # Best threshold is at midpoint
                    best_thr = (thresholds[j] + thresholds[j - 1]) / 2
                    
        return best_i, best_thr

    def fit(self, X, y):
        '''
        Build decision tree (recursively find best spilt)
        '''
        self.num_classes = len(set(y))
        self.num_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.num_classes))

    def _entropy(self, theta):
        return -theta * math.log(theta) - (1 - theta) * math.log(1 - theta)

    def _grow_tree(self, X, y, depth=0):
        '''
        Helper function for `fit`
        '''
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            i, thr = self._best_split(X, y)
            if i is not None:
                idc_left = X[:, i] < thr
                X_left, y_left = X[idc_left], y[idc_left]
                X_right, y_right = X[~idc_left], y[~idc_left]
                node.feature_index = i
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def predict(self, X):
        return [self._predict(features) for features in X]

    def _predict(self, inputs):
        '''
        Return predictions for all labels in X by traversing the decision tree,
        going left if feature value is below threshold and right otherwise
        '''
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

def initialize_data(filename):
    inputs = []
    targets = []
    with open(filename, mode='r') as f:
        for line in f.readlines():
            line_arr = line.split()
            targets.append(int(line_arr[0]))
            line_arr = line_arr[1:]
            feat_arr = [0 for i in range(129)]
            for elem in line_arr:
                pair_values = elem.split(':')
                feat_arr[int(pair_values[0])] = int(pair_values[1])
            inputs.append(feat_arr)
    return np.array(inputs), np.array(targets)


# Build X (list of feature values) and y (list of classifications)
X, y = initialize_data('a4a.txt')

clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)

# Test decision tree
inputs, targets = initialize_data('a4a_test.txt')
pred = clf.predict(inputs)
correct = 0
for i, p in enumerate(pred):
    print('Predicted: {}'.format(p), 'Actual: {}'.format(targets[i]))
    if p == targets[i]:
        correct += 1
print('len(targets):', len(targets))
print('Accuracy: ', correct / 27780)