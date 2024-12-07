#### User Invariant Preference Learning for Multi-Behavior Recommendation

This is an implementation of UIPL on Python 3



#### Getting Started

First, run the data_process.py file in data/xxx to collate the data set.

then, please create a new folder named 'check_point' to save the check point file.

you can run Tmall dataset with:

`python3 main.py --data_name tmall`

or

`./b_tmall.sh`


The optimal hyperparameters for the model are shown in the table below:

| Dataset | log_reg (&lambda;) | ort_reg(&alpha;) | nce_reg(&beta;) | kn_reg(&gamma;) | reg_weight | temperature(&tau;) |  lr  |
|---------|:------------------:|:----------------:|:---------------:|:---------------:|:----------:|:------------------:|:----:|
| Tmall   |        0.01        |       1e-4       |       0.1       |       0.1       |    1e-3    |        0.1         | 1e-3 |
| Taobao  |        0.01        |       1e-4       |      0.01       |      0.01       |    1e-3    |        0.1         | 1e-3 |
| Yelp    |        0.1         |       1e-3       |      0.01       |       0.1       |    1e-3    |        0.5         | 1e-3 |
| ML10M   |        0.5         |       1e-3       |       0.1       |       0.1       |    1e-3    |        0.3         | 1e-3 |
