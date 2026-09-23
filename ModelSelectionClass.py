import pandas as pd

from sklearn.metrics import accuracy_score

class ModelSelection():

    def __init__(self, grids, grid_dict):
        self.grids = grids
        self.grid_dict = grid_dict
        self.best_models_data = []
        self.best_scores = {}

    def choose(self, X_train, y_train, X_valid, y_valid):
        for i, grid in enumerate(self.grids):
            model = self.grid_dict[i]
            print(f"Estimator: {model}")

            grid.fit(X_train, y_train)
            print(f"Best params: {grid.best_params_}")
            print(f"Best training accuracy: {grid.best_score_:.3f}")

            valid_pred = grid.best_estimator_.predict(X_valid)
            valid_score = accuracy_score(y_valid, valid_pred)
            print(f"Validation set accuracy score for best params: {valid_score:.3f}")

            self.best_models_data.append({
                "model": model,
                "params": grid.best_params_,
                "valid_score": valid_score
            })

            self.best_scores[model] = valid_score

        best_model_name = max(self.best_scores, key=self.best_scores.get)
        print(f"Classifier with best validation set accuracy: {best_model_name}")

        best_index = [k for k, v in self.grid_dict.items() if v == best_model_name][0]
        best_grid = self.grids[best_index]
        return best_grid.best_estimator_

    def best_results(self):

        if self.best_models_data:
            results = pd.DataFrame(self.best_models_data)
            return results
        else:
            print("Перед работой данного метода нужно запустить метод choose.")
