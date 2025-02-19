# Data

Please follow these steps to add your task data in this folder.

1. Create a folder for your dataset. Please follow `kebab-case` while naming your folder.

2. Add your baseline, evaluation and data files following the directory structure given below.

```
_helpers/ - helper functions to prepare the dataset (Optional)
data/
    train.csv
    valid.csv
    test.csv
baseline.py
evaluate.py
sample_submission.csv (Optional)
```

3. If your baseline code is a collection of scripts, please create a folder named `baseline` and save all the baseline codebase in that folder EXCEPT `evaluate.py`.
