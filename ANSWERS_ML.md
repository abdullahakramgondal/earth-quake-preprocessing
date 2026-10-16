# Day 2-3 Lab: ML Answers



Note: the live feed changes every minute, so the number of rows differs a little between runs (about 1,950 to 2,015 rows). The experiments in Q1 (magType), Q4 (sig) and Q5 (polynomial) were run at different times, so each comparison uses numbers from the same run, and small differences between runs are normal.

---

## 1. Baseline

**Evidence** (`python -m quakes.train --task regression --model linear`)

| metric | baseline (predict the mean) | linear model |
|---|---|---|
| MAE | 0.9321 | 0.3945 |
| RMSE | 1.2651 | 0.5133 |
| R2 | -0.0023 | 0.8350 |

5-fold CV R2 on train: 0.8404 +/- 0.0216

Extra check (earlier run, 2,015 rows): with magType the test R2 was 0.8309; after removing magType from config.NOMINAL it was 0.8289 (CV 0.8265). Almost the same.

**Answer**
The linear model beats the "predict the mean" baseline by a lot: RMSE drops from 1.27 to 0.51 and R2 goes from about 0 to 0.835. The CV score (0.84) matches the test score, so the result is stable. However, the handout says to expect a modest R2 after removing leaky columns, and 0.83 is quite high. I checked magType and it is not the reason, because removing it did not change the score. Columns like nst, dmin, rms and gap may still carry information linked to the magnitude. So it is a good first model, but I would check these columns more before deploying it.

In simple words: the baseline always guesses the average magnitude and is wrong by about 1.27 on average (RMSE). My model is wrong by about 0.51, so it is much better. R2 0.83 means it explains about 83% of the variation. Because 0.83 looked too good, I tested magType by removing it, and the score stayed the same, so magType is not leaking.

---

## 2. Accuracy paradox

**Evidence** (`python -m quakes.train --task classification --model logistic`, threshold 0.5)

| metric | dummy baseline | logistic model |
|---|---|---|
| accuracy | 0.9133 | 0.9643 |
| precision | 0.0000 | 0.7778 |
| recall | 0.0000 | 0.8235 |
| f1 | 0.0000 | 0.8000 |
| roc_auc | 0.5000 | 0.9896 |

**Answer**
The dummy classifier gets accuracy 0.913 and recall 0.0. It always predicts "not a big quake", and only about 9% of events are big quakes, so it is right most of the time but never finds a single big quake. Accuracy is the wrong headline metric because the classes are imbalanced. Recall and precision show whether the model finds the rare events.

In simple words: if only 9 out of 100 quakes are big and a model says "not big" every time, it is correct 91 times, so accuracy looks great, but it catches zero big quakes. That is why I look at recall and precision.

---

## 3. Threshold

**Evidence** (same classification run)

| | threshold 0.5 | threshold 0.20 (chosen on train) |
|---|---|---|
| accuracy | 0.9643 | 0.9643 |
| precision | 0.7778 | 0.7273 |
| recall | 0.8235 | 0.9412 |
| f1 | 0.8000 | 0.8205 |

5-fold CV ROC-AUC on train: 0.9917 +/- 0.0034

**Answer**
My search chose 0.20, which is lower than 0.5. I chose it from out-of-fold predictions on the training set only (cross_val_predict) and applied it once on the test set. Recall went up from 0.824 to 0.941 and precision went down from 0.778 to 0.727. In an earthquake alert system I would rather make a false alarm than miss a big quake, because a missed quake can cost lives and a false alarm only costs some trouble.

In simple words: the model gives a probability. At 0.5 it says "big" only if it is more than 50% sure. At 0.20 it warns more often, so it catches more big quakes but also gives more false alarms. I picked the threshold from training data, not test data, so the test score is honest.

---

## 4. Leakage check

**Evidence** (I moved `sig` from `config.LEAKY` into `config.NUMERIC`, deleted the database, reran ETL and training. The "with sig" run used 1,964 rows, the "without sig" numbers are from today's run with 1,957 rows.)

Regression:

| metric | without sig | with sig |
|---|---|---|
| MAE | 0.3945 | 0.2990 |
| RMSE | 0.5133 | 0.4150 |
| R2 | 0.8350 | 0.9024 |

CV R2 on train with sig: 0.8141 +/- 0.1777

Classification:

| metric | without sig | with sig |
|---|---|---|
| precision | 0.7778 | 0.9429 |
| recall | 0.8235 | 0.9706 |
| f1 | 0.8000 | 0.9565 |
| roc_auc | 0.9896 | 0.9996 |

**Answer**
When I put sig back in, the scores got better: regression R2 went from 0.835 to 0.902 and classification ROC-AUC from 0.990 to 0.9996. But the better number is a lie. sig is calculated from the magnitude, so the model is partly reading the answer, and at prediction time we would not have it. The honest scores are the ones without sig. The CV std of 0.18 with sig is also very large, which shows the result is not stable.

In simple words: sig is a significance score that USGS computes using the magnitude. Giving it to the model is like giving it the answer sheet. The score goes up, but when a quake has just happened we do not have sig yet, so the improvement is fake.

---

## 5. Complexity

**Evidence** (`--model poly --degree 1 / 2 / 4`, earlier run with 2,015 rows; train R2 from a separate check with the same split)

| degree | train R2 | test R2 | CV R2 on train |
|---|---|---|---|
| 1 | 0.8427 | 0.8314 | 0.8314 |
| 2 | 0.8952 | 0.8401 | 0.8313 |
| 4 | 0.9719 | -136903.69 | -201121.50 |

At degree 4 sklearn also printed "LinAlgWarning: Ill-conditioned matrix".

**Answer**
Degree 1: train 0.843 and test 0.831 are almost equal. Degree 2: train 0.895 and test 0.840, so a gap starts, but test is still slightly better than degree 1. Degree 4: train 0.972 but test R2 is about -136,903. So test performance falls below the training result a little at degree 2 and completely at degree 4. This is overfitting: a high-degree model memorises the training data, and too many polynomial features also make it numerically unstable. The best choice here is degree 1 or 2.

In simple words: a higher degree makes the model more flexible. On train data it keeps improving (0.84, 0.90, 0.97), but that is memorising. On test data degree 2 is only a little better and degree 4 completely fails, so the gap between train and test shows where overfitting starts.

---

## 6. Gradient descent

**Evidence** (from the notebook, Section 4)

- GD coefficients: [1.956, -0.989, 0.52]
- sklearn LinearRegression coefficients: [1.956, -0.989, 0.52]
- Unscaled data: lr=1e-4, 500 iterations, final loss 6.530
- Scaled data: lr=0.1, 500 iterations, final loss 0.130

**Answer**
Yes. On the notebook data my GDLinearRegression coefficients match LinearRegression to three decimals. Without scaling, I had to use a tiny learning rate (1e-4), because the feature with a big scale makes larger steps diverge. After 500 iterations the loss was still 6.53, against 0.13 for the scaled data. So scaling makes gradient descent converge faster and allows a bigger learning rate.

In simple words: gradient descent walks downhill in steps. If one feature has a huge scale, the slope is very steep in that direction, so a normal step jumps over the minimum and the loss explodes. After scaling, all features are on the same scale, so one learning rate of 0.1 works.