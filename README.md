# Titanic Survival Prediction — Kaggle ML Project

A portfolio machine learning project based on the Kaggle Titanic dataset. The goal is to predict passenger survival using structured tabular data.

## Project overview

This project demonstrates an end-to-end classical ML workflow:

- loading and inspecting train/test data;
- checking missing values;
- feature engineering from names, tickets, cabins, and family columns;
- preprocessing numeric and categorical features with `scikit-learn` pipelines;
- comparing several tree-based models with stratified cross-validation;
- training a final model;
- generating `submission.csv` for Kaggle.

## Dataset

The dataset is from the Kaggle Titanic competition.

Expected files:

```text
data/train.csv
data/test.csv
```

The CSV files are not included in this repository by default. Download them from Kaggle and place them in the `data/` folder.

## Repository structure

```text
.
├── data/
│   └── .gitkeep
├── notebooks/
│   └── titanic_survival_prediction.ipynb
├── outputs/
│   └── .gitkeep
├── src/
│   └── features_and_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Features used

Original useful columns:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`

Engineered features:

- `Title` — extracted from passenger name;
- `FamilySize` — `SibSp + Parch + 1`;
- `IsAlone` — whether the passenger travelled alone;
- `SmallFamily` — family size between 2 and 4;
- `LargeFamily` — family size of 5 or more;
- `CabinKnown` — whether cabin information is available;
- `Deck` — first letter of the cabin;
- `TicketGroupSize` — number of passengers sharing the same ticket;
- `FarePerPerson` — fare divided by family size.

## Models compared

The notebook compares:

- `GradientBoostingClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`

Validation is done with `StratifiedKFold` cross-validation and `accuracy` scoring.

## How to run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put `train.csv` and `test.csv` into the `data/` folder.
4. Open and run:

```text
notebooks/titanic_survival_prediction.ipynb
```

5. The notebook will create:

```text
outputs/submission.csv
```

## Notes

This is a clean portfolio baseline. Further improvements could include hyperparameter tuning with `GridSearchCV`, more detailed exploratory data analysis, and model interpretation using feature importance.
