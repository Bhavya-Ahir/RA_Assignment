# DNN Hyperparameter Tuning Prediction

## Introduction
This project aims to predict the final accuracy of a Deep Neural Network (DNN) based on early training results, reducing the time needed for hyperparameter tuning.


## Files
```
.
├── data/                 # Directory containing data and files
├── models/               # Directory containing trained models and results
├── scripts/              
│   ├── train.py          # Script for training models
│   ├── evaluate.py       # Script for evaluating models
│   ├── utils.py          # Utility functions for data loading and preprocessing
├── README.md             # Project documentation
├── requirements.txt      # List of dependencies
└── .gitignore            # Git ignore file to exclude unnecessary files
```


## Installing rerquirements
```
pip install -r requirements.txt
```

# To train and save the models run below
```
python scripts/train.py
```

# To use the pre-saved models and re-create result run below, results would be added in csv format under /models dir
```
python scripts/evaluate.py
```