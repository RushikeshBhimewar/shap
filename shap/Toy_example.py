import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

np.random.seed(101)
X_train = pd.DataFrame(
                      {'x': list(np.random.randint(0, 100, size= 10)),
                      'y': list(np.random.randint(100, 200, size=10)),
                      'z': list(np.random.randint(0, 200, size=10))}
                      )
y_train = pd.Series(list(np.random.randint(100,150,size=10)))
y_train.name = 't'

# print(X_train)
# print(y_train)

tree_model = DecisionTreeRegressor(criterion="absolute_error", max_depth=2, min_samples_leaf=1, min_samples_split=2, random_state = 100)

tree_model.fit(X=X_train, y=y_train)

X_test = pd.DataFrame({'x':[75],'y':[124],'z':[215]})

train_pred = tree_model.predict(X_train)
test_pred = tree_model.predict(X_test)


expla_perm = shap.Explainer(tree_model.predict, X_train, algorithm = "permutation")
shap_values_perm = expla_perm(X_train,main_effects=True)

print(shap_values_perm.values)