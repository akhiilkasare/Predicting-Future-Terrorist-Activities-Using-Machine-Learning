import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
import pickle


def run(fold):

    df = pd.read_csv('train_folds.csv')
    # df = df.drop('type', axis=1)
    # df = df.fillna(0)

    # Training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df[['weaptype1','property','iyear','extended','targtype1','ishostkid',
      'iday', 'provstate','city','imonth','region','gname','country', 'nwound',
      'natlty1', 'doubtterr', 'success','suicide']].values
    y_train = df['attacktype1'].values

    # Repeating the same steps for the validation dataset
    x_valid = df[['weaptype1','property','iyear','extended','targtype1','ishostkid',
      'iday', 'provstate','city','imonth','region','gname','country', 'nwound',
      'natlty1', 'doubtterr', 'success','suicide']].values
    y_valid = df['attacktype1'].values

    # Initializing a simple Random Forest Classifier 
    clf = ensemble.RandomForestClassifier()

    # Fitting the model on the training data
    clf.fit(x_train, y_train)

    # Create predictions for validation samples
    preds = clf.predict(x_valid)

    # Calculating the accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # Saving the model
    joblib.dump(clf, f"/home/akhil/Downloads/machine_learning/gtd/models/rf_{fold}.pkl")

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
     