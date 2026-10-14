# Disneyland Reviews: Which Park Is This Review About?

An NLP text-classification project. It predicts **which Disneyland park** (California, Paris or Hong Kong) a visitor review refers to, using only the review text, its length and the star rating.

The project compares **7 models** that combine different text representations (Bag of Words, TF-IDF, word embeddings) with different classifiers (Naive Bayes, Linear SVM, Logistic Regression). It also handles class imbalance with SMOTE and balanced-accuracy evaluation, and explains *why* some models outperform others.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anjaperovic22/disneyland-reviews-nlp/blob/main/disneyland_reviews_nlp.ipynb)

> Academic project for the *Advanced Data Analysis* course at the Faculty of Organizational Sciences, University of Belgrade (2026).

## Why this problem?

Visitors to different parks write differently. They care about different things (queues, food, park size), use different vocabulary, and rate differently. Detecting the location from the text alone helps when:
- segmenting customer feedback by region when location metadata is missing or unreliable;
- finding market-specific complaints. For example, the most frequent phrases in negative Paris reviews were *“fast pass”*, *“long queues”* and *“rides closed”*.

## Dataset

[Disneyland Reviews](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) (Kaggle): **42,117** TripAdvisor reviews of three parks.

| Class | Share |
|---|---|
| California | 46.1 % |
| Paris | 31.1 % |
| Hong Kong | 22.8 % |

The classes are **imbalanced**, so **balanced accuracy** is the main evaluation metric.
For faster experimentation, a **stratified subsample of 6,000 reviews** keeps the original class proportions. It is split 80/20 into train and test sets, also stratified.

## Approach

1. **Exploratory analysis**: class distribution, review length and rating per park, and the most frequent bigrams in negative reviews.
2. **Text preprocessing with spaCy**: lemmatisation and removal of punctuation, numbers, URLs and stop words. **Negations** (*no, not, never*) are deliberately kept, because removing them can flip the meaning of a review.
3. **Features**: the review text plus two numeric features, review length and rating, combined with a `ColumnTransformer`.
4. **Models**: all built as scikit-learn / imbalanced-learn `Pipeline`s.
5. **Tuning**: `GridSearchCV` with 5-fold cross-validation, optimising balanced accuracy.
6. **Evaluation**: confusion matrices, per-class precision, recall and F1, and balanced accuracy for every model.

## Results

| Model | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **BoW + Multinomial NB (tuned, `min_df=0.005`)** | **0.816** | **0.809** | **0.819** |
| TF-IDF + Logistic Regression | 0.797 | 0.789 | 0.797 |
| BoW + Multinomial NB (baseline) | 0.789 | 0.781 | 0.793 |
| TF-IDF + Linear SVM + SMOTE | 0.784 | 0.780 | 0.787 |
| TF-IDF + Linear SVM | 0.767 | 0.772 | 0.782 |
| Word embeddings (spaCy `en_core_web_lg`) + Linear SVM | 0.765 | 0.775 | 0.784 |
| TF-IDF + Multinomial NB | 0.678 | 0.675 | 0.691 |

### Key findings
- **A well-tuned simple model wins.** Tuning `min_df` raised Bag of Words + Naive Bayes from 0.789 to **0.816** balanced accuracy. The tuned value drops words that appear in fewer than ~30 reviews, which removes typos and noise.
- **TF-IDF hurts Naive Bayes** (0.678). TF-IDF boosts rare terms, and an analysis of the rarest unigrams and bigrams showed these are mostly one-off words and typos. Naive Bayes then relies on statistically unreliable signals.
- **SMOTE helps the minority class.** Oversampling raised Hong Kong recall from 0.70 to 0.78 and balanced accuracy from 0.767 to 0.784.
- **Logistic Regression is the most balanced** across classes. It needed scaled numeric features and the `saga` solver to converge.
- **The data itself sets a limit.** All three parks share much of their vocabulary (*park, ride, time, day, wait*), which explains most of the remaining confusion, especially Paris being predicted as California.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · spaCy (`en_core_web_sm`, `en_core_web_lg`) · Matplotlib · Seaborn · Google Colab

## How to run

The easiest way is to click **Open in Colab** above and run all cells. When prompted, upload `DisneylandReviews.csv`, which you can download from the Kaggle link above.

To run locally:

```bash
pip install pandas numpy scikit-learn imbalanced-learn spacy matplotlib seaborn jupyter
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
jupyter notebook disneyland_reviews_nlp.ipynb
```

Locally, replace the Colab upload cell with `pd.read_csv("DisneylandReviews.csv", encoding="latin-1")`.
Some steps (spaCy tokenisation inside the vectorizers, the SVM with SMOTE, the grid search) take several minutes.

## Author

**Anja Perović**, student at the Faculty of Organizational Sciences, University of Belgrade
