# Who will say yes? A term-deposit propensity model

We aim to predict which bank clients will subscribe to a term deposit after a phone campaign.

**Status:** Baselines done. Gradient boosting, confidence intervals, calibration and lift table are in progress.

## The question

A bank has limited advisor time, so before calling anyone, we wish to rank clients by how likely they are to subscribe.
We want to see if calling the top part of the list will reach most of the eventual subscribers.

## Data

The data used in this study can be found in [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`).
It consists of 41,188 phone contacts by a Portuguese bank during 2008-2010.
About 11% of those subscribed.
The dataset contains client attributes, contact history, and five macroeconomic indicators (employment variation, consumer price index, consumer confidence, 3-month Euribor, and number employed).

## Key decisions

- **No leakage.** Longer calls often signal a subscription, but `duration` (call length) is only known after the call. Hence, we exclude it.
- **Time-based split.** The file is in chronological order. The model is trained on the first 80% of rows and tested on the last 20%. The test period has a much higher subscription rate than training (30.8% vs 6.4%), which makes this a deliberately hard test.
- **Ranking metrics.** ROC AUC and PR AUC, read against each part's own base rate. Accuracy is not used, because the dumbest possible model (which always predicts no) already scores 69%.
- **Unknown values kept.** We put them in their own category.
- **`pdays`** (number of days before last contact). A value of 999 indicates "never contacted before", so we split the data into a "contacted before" column and a `pdays` column with 999 replaced by 0.
- **Preprocessing inside a scikit-learn Pipeline.** This ensures that nothing is learned from the test set.

## Results so far

The test set is reported in two parts: the end of the second year in the file (block 1 tail) and the whole third year (block 2, a new period).

| Test view | Rows | Base rate | ROC AUC | PR AUC | ROC AUC without macro features |
|---|---|---|---|---|---|
| Full test | 8,236 | 0.308 | 0.748 | 0.527 | 0.700 |
| Block 1 tail | 6,178 | 0.237 | 0.733 | 0.439 | 0.672 |
| Block 2 | 2,058 | 0.521 | 0.654 | 0.632 | 0.645 |

Logistic regression; the majority-class baseline scores ROC AUC 0.5 everywhere. Confidence intervals will be added with the final evaluation.

## What I learned

- **The model ranks clients better than chance, but less in a new period.** ROC AUC drops from 0.73 to 0.65 in block 2.
- **Scores are not calibrated probabilities.** The model underpredicts (average 0.41 vs 0.52 actual in block 2), because the subscription rate rose over time.
- **Macroeconomic features partly act as a clock.** Within a single month (May, 3,274 clients) they vary very little, and ROC AUC is about 0.59 with or without them. Hence, a (likely) significant part of the ranking skill comes from telling periods apart instead of the actual clients.

## Limits

One bank, one campaign period, correlation not causation. This project does not cover large-scale infrastructure, Databricks or generative AI.

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
src/         data loading and splitting helpers
sql/         first SQL queries
notebooks/   01_eda, 02_baselines
```

*Data: Moro, Cortez and Rita (2014), "A data-driven approach to predict the success of bank telemarketing", Decision Support Systems.*
