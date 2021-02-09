import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # Loading the csv file
    df = pd.read_csv("cleaned_data.csv")

    # We create a new column called kfold and fill it with -1
    df['kfold'] = -1

    # The next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Fetch targets
    X = df[['weaptype1','property','iyear','extended','targtype1','ishostkid',
      'iday', 'provstate','city','imonth','region','gname','country', 'nwound',
      'natlty1', 'doubtterr', 'success','suicide']]
   
    y = df['attacktype1']

    # Initiate the kfold class for the model-selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Filling the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f 

    # Saving the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)
