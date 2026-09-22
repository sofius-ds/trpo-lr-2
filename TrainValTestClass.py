from sklearn.model_selection import train_test_split

class TrainValidationTest():

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split(self):
        X_split, X_test, y_split, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=21, stratify=self.y)
        X_train, X_valid, y_train, y_valid = train_test_split(X_split, y_split, test_size=0.2, random_state=21, stratify=y_split)

        return X_train, X_valid, X_test, y_train, y_valid, y_test
