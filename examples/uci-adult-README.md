# UCI Adult test data

- `uci-adult-income-5000.csv`: 5,000 labeled rows for a Workbench classification project.
- `uci-adult-income-batch-250.csv`: 250 feature-only rows for Batch prediction.
- Target column: `income` (`<=50K` or `>50K`).
- Mixed numeric/categorical features and intentionally preserved missing values exercise the preprocessing path.
- Rows were deterministically shuffled with seed 42, then sampled from `adult.data`.

Source: Becker, B. & Kohavi, R. (1996), *Adult*, UCI Machine Learning Repository,
https://doi.org/10.24432/C5XW20. The source dataset is licensed under CC BY 4.0.

