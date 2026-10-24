# Who will say yes? A term-deposit propensity model

We aim to predict which bank clients will subscribe to a term deposit after a phone campaign.

**Status:** Complete. Baselines, model comparison, evaluation with bootstrap confidence intervals, calibration, gains table, and within-period checks are done.

## The question

A bank has limited advisor time, so before calling anyone, we wish to rank clients by how likely they are to subscribe.
We want to see if calling the top part of the list will reach most of the eventual subscribers.

**Short answer:** partly, and it depends on the period. In the period right after training, calling the top 20% of the list reaches about 40% of subscribers (twice as many as random calling). However, in the newest period, it only reaches 25%.

## Data

The data used in this study can be found in [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`).
It consists of 41,188 phone contacts by a Portuguese bank during 2008-2010.
About 11% of those subscribed.
The dataset contains client attributes, contact history, and five macroeconomic indicators (employment variation, consumer price index, consumer confidence, 3-month Euribor, and number employed).

## Key decisions

- **No leakage.** Longer calls often signal a subscription, but `duration` (call length) is only known after the call. Hence, we exclude it.
- **Time-based split.** The file is in chronological order. The model is trained on the first 80% of rows and tested on the last 20%. The test period has a much higher subscription rate than training (30.8% vs 6.4%), which makes this a deliberately hard test.
- **Time-aware model selection.** Models are compared with expanding-window cross-validation (5 folds) inside the training data, so the future is never used to predict the past. The test set is only used once, in the final evaluation.
- **Ranking metrics.** ROC AUC and PR AUC, read against each part's own base rate. Accuracy is not used, because the dumbest possible model (which always predicts no) already scores 69%.
- **Duplicates removed.** This only affected 12 rows of the 41.188.
- **Uncertainty.** 95% bootstrap confidence intervals (1,000 resamples of the test set).
- **Unknown values kept.** We put them in their own category.
- **`pdays`** (number of days before last contact). A value of 999 indicates "never contacted before", so we split the data into a "contacted before" column and a `pdays` column with 999 replaced by 0.
- **Preprocessing inside a scikit-learn Pipeline.** This ensures that nothing is learned from the test set.

## Results

The test set is reported in two parts: the end of the second year in the file (block 1 tail) and the whole third year (block 2, a new period).

ROC AUC on the test set, with 95% bootstrap intervals (the majority-class baseline scores 0.5 everywhere):

| Model | Full test (8,236 rows) | Block 1 tail (6,178) | Block 2 (2,058) |
|---|---|---|---|
| Base rate | 0.308 | 0.237 | 0.521 |
| Logistic regression, with macro features | 0.748 [0.736, 0.759] | 0.733 [0.718, 0.747] | 0.654 [0.633, 0.678] |
| Logistic regression, without macro features | 0.700 [0.688, 0.713] | 0.672 [0.657, 0.688] | 0.645 [0.621, 0.669] |
| Gradient boosting, with macro features | 0.612 [0.598, 0.626] | 0.658 [0.642, 0.672] | 0.570 [0.543, 0.598] |
| Gradient boosting, without macro features | 0.659 [0.646, 0.672] | 0.609 [0.592, 0.625] | 0.643 [0.618, 0.667] |

For the best model (logistic regression with macro features), PR AUC is 0.527 against a base rate of 0.308 (full test), 0.439 against 0.237 (block 1 tail), and 0.632 against 0.521 (block 2).

Share of subscribers reached by calling the top of the list (same model):

| Test view | Top 10% of list | Top 20% of list |
|---|---|---|
| Block 1 tail | 21.0% (lift 2.1) | 40.5% (lift 2.0) |
| Block 2 | 10.8% (lift 1.1) | 25.0% (lift 1.2) |

Model selection inside the training data: with time-aware cross-validation all models have a mean ROC AUC of only 0.51 to 0.54 (fold to fold, 0.42 to 0.66). Ordinary shuffled cross-validation gives 0.67 for logistic regression, so it would have overstated the skill.

## What I learned

- **The model ranks clients better than chance, but less in a new period.** ROC AUC drops from 0.73 to 0.65 in block 2, and the confidence intervals do not overlap.
- **Logistic regression beat gradient boosting.** A likely reason is that trees cannot extrapolate: the test period has interest rates far below anything seen in training (minimum 0.63% vs 1.30%), and boosting's predictions stayed close to the training subscription rate (about 8% vs an actual 31%).
- **Scores are not calibrated probabilities.** The model underpredicts (average 0.41 vs 0.52 actual in block 2), because the subscription rate rose over time. It would need recalibrating on recent data before the scores are used as probabilities.
- **The top of the list is unreliable in a new period.** In block 2 the second to fourth deciles have a higher subscription rate than the top decile.
- **Macroeconomic features partly act as a clock.** They help logistic regression in the block 1 tail (0.733 vs 0.672) but not in block 2 (0.654 vs 0.645), and within a single month (May, 3,274 clients) they make no difference (about 0.59 with or without them).
- **Within a single month, the model still ranks clients.** ROC AUC is between 0.56 and 0.68 per month in block 1 and between 0.63 and 0.80 in block 2. Ranking by the outcome of the previous campaign (`poutcome`) alone already gets most of the way there (average 0.59 and 0.68 against 0.61 and 0.72 for the model).

## Limits

One bank, one campaign period, correlation not causation. The test period differs strongly from the training period, so the results are a stress test and not a forecast of future performance. The project stops at evaluation: there is no deployment, monitoring, or retraining.

## Run it

```
conda create -n bank-propensity python=3.14
conda activate bank-propensity
pip install -r requirements.txt
```

Download the CSV into `data/raw/`, then:

```
python src/load_data.py
```

and open the notebooks in order.

```
src/         data loading, features, and model helpers
sql/         first SQL queries
notebooks/   01_eda, 02_baseline, 03_models, 04_evaluation
```

*Data: Moro, Cortez and Rita (2014), "A data-driven approach to predict the success of bank telemarketing", Decision Support Systems.*