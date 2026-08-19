# Naive Bayes Classifier - Machine Learning

A comprehensive repository for learning and implementing the **Naive Bayes Classifier** from fundamental probability concepts to practical machine learning applications using **Python** and **Scikit-learn**.

This repository covers the mathematical intuition behind Naive Bayes, Bayes' Theorem, conditional probability, different Naive Bayes variants, data preprocessing, model training, prediction, evaluation metrics, hyperparameter tuning, and practical classification workflows.

---

## 📚 About the Project

**Naive Bayes** is a family of probabilistic classification algorithms based on **Bayes' Theorem**.

The algorithm calculates the probability of a class given a set of input features and predicts the class with the highest posterior probability.

Naive Bayes is called **"naive"** because it assumes that features are **conditionally independent** given the class.

Despite this simplifying assumption, Naive Bayes performs remarkably well on many real-world classification problems, especially **text classification**.

---

## 🎯 Learning Objectives

By completing this repository, you will understand:

* What Naive Bayes is
* Why Naive Bayes is used
* Bayes' Theorem
* Conditional probability
* Prior probability
* Likelihood
* Posterior probability
* The Naive Bayes assumption
* Different types of Naive Bayes classifiers
* Data preprocessing
* Model training
* Model prediction
* Model evaluation
* Confusion Matrix
* Precision, Recall, and F1 Score
* Scikit-learn implementation
* Hyperparameter tuning
* Cross-validation
* Practical classification problems

---

## 📖 Topics Covered

### 1. Introduction to Classification

* What is Classification?
* Supervised Learning
* Classification vs Regression
* Binary Classification
* Multiclass Classification
* Real-world classification problems

---

### 2. Probability Fundamentals

Before understanding Naive Bayes, it is important to understand basic probability concepts.

#### Probability

Probability measures how likely an event is to occur.

```text
P(A)
```

#### Conditional Probability

The probability of event `A` occurring given that event `B` has already occurred.

```text
P(A | B)
```

#### Joint Probability

The probability of two events occurring together.

```text
P(A, B)
```

#### Prior Probability

The probability of a class before observing the features.

```text
P(C)
```

#### Likelihood

The probability of observing the features given a class.

```text
P(X | C)
```

#### Posterior Probability

The probability of a class after observing the features.

```text
P(C | X)
```

---

## 🧮 Bayes' Theorem

Naive Bayes is based on **Bayes' Theorem**:

```text
P(C | X) = [P(X | C) × P(C)] / P(X)
```

Where:

* `P(C | X)` = Posterior probability
* `P(X | C)` = Likelihood
* `P(C)` = Prior probability
* `P(X)` = Evidence

The classifier predicts the class with the **highest posterior probability**.

---

## 🧠 Naive Bayes Assumption

For multiple features:

```text
X = X₁, X₂, X₃, ..., Xₙ
```

Naive Bayes assumes:

```text
P(X | C) =
P(X₁ | C) × P(X₂ | C) × ... × P(Xₙ | C)
```

Therefore:

```text
P(C | X₁, ..., Xₙ) ∝
P(C) × ∏ P(Xᵢ | C)
```

The **conditional independence assumption** is the key idea behind Naive Bayes.

---

# 🔢 Types of Naive Bayes Classifiers

## 1. Gaussian Naive Bayes

Gaussian Naive Bayes is generally used when continuous numerical features approximately follow a **Gaussian (normal) distribution**.

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
```

### Common Applications

* Medical classification
* Sensor data
* Numerical datasets
* General classification problems

---

## 2. Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for **discrete count-based features**, especially in text classification.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
```

### Common Applications

* Text classification
* Spam detection
* Document classification
* Sentiment analysis

---

## 3. Bernoulli Naive Bayes

Bernoulli Naive Bayes is used for **binary or boolean features**.

```python
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
```

### Common Applications

* Binary text features
* Word presence/absence
* Boolean datasets
* Binary feature classification

---

## 4. Complement Naive Bayes

Complement Naive Bayes is particularly useful for **imbalanced text classification problems**.

```python
from sklearn.naive_bayes import ComplementNB

model = ComplementNB()
```

---

## ⚖️ Gaussian vs Multinomial vs Bernoulli vs Complement

| Algorithm     | Feature Type                  | Common Use                     |
| ------------- | ----------------------------- | ------------------------------ |
| GaussianNB    | Continuous numerical features | Numerical classification       |
| MultinomialNB | Counts / Frequencies          | Text classification            |
| BernoulliNB   | Binary features               | Boolean / binary data          |
| ComplementNB  | Count-based features          | Imbalanced text classification |

---

# 🔄 Naive Bayes Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Preprocessing
   ↓
Naive Bayes Model
   ↓
Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Final Model
```

---

# 📊 Data Preprocessing

Important preprocessing steps may include:

* Handling missing values
* Removing duplicates
* Detecting outliers
* Encoding categorical variables
* Feature scaling when appropriate
* Text preprocessing
* Train-test splitting

### Train-Test Split Example

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 🧪 Gaussian Naive Bayes Using Scikit-learn

Basic implementation:

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 📝 Multinomial Naive Bayes for Text Classification

For text classification, text needs to be converted into numerical features.

A common approach is **TF-IDF**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
```

---

# 🔗 Naive Bayes Pipeline

A Scikit-learn `Pipeline` can combine preprocessing and model training.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 📈 Model Evaluation

Classification models can be evaluated using multiple metrics.

## Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

## Precision

```text
Precision = TP / (TP + FP)
```

## Recall

```text
Recall = TP / (TP + FN)
```

## F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

# 📊 Confusion Matrix

A confusion matrix shows how many predictions fall into each category.

```text
                    Predicted
                  Positive  Negative

Actual Positive     TP        FN

Actual Negative     FP        TN
```

### Example

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

---

# 🧾 Classification Report

Scikit-learn provides a convenient classification report.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

---

# 🎯 Accuracy Score

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

---

# 🔍 Probability Prediction

Naive Bayes models can also return class probabilities.

```python
probabilities = model.predict_proba(X_test)

print(probabilities)
```

The predicted class is generally the class with the highest probability.

---

# ⚠️ Laplace Smoothing

A common problem in Naive Bayes occurs when a feature has zero probability for a particular class.

If:

```text
P(Xᵢ | C) = 0
```

then multiplying probabilities can make the entire posterior probability zero.

**Laplace smoothing** helps solve this problem by assigning a small non-zero probability.

For Multinomial Naive Bayes in Scikit-learn:

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0)
```

The `alpha` parameter controls smoothing.

---

# 🎛️ Hyperparameter Tuning

Important Naive Bayes hyperparameters include:

### MultinomialNB

* `alpha`
* `fit_prior`

### BernoulliNB

* `alpha`
* `binarize`
* `fit_prior`

### GaussianNB

* `var_smoothing`

---

## GridSearchCV Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

param_grid = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
```

---

# 🔄 Cross-Validation

Cross-validation can be used to obtain a more reliable estimate of model performance.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("CV Scores:", scores)
print("Mean Accuracy:", scores.mean())
```

---

# 🌍 Real-World Applications

Naive Bayes is widely used in:

* Spam Email Detection
* Sentiment Analysis
* Text Classification
* Document Classification
* News Categorization
* Medical Classification
* Recommendation Systems
* Customer Feedback Classification
* Language Detection
* Fraud Detection
* Search and Information Retrieval

---

# 💡 Advantages

* Simple and easy to implement
* Fast training and prediction
* Works well with high-dimensional data
* Effective for text classification
* Requires relatively little training data
* Supports multiclass classification
* Produces probability estimates
* Computationally efficient

---

# ⚠️ Disadvantages

* The feature independence assumption is often unrealistic
* Can be affected by unseen feature combinations
* Probability estimates may not always be well calibrated
* Performance can suffer when feature dependencies are important
* Different Naive Bayes variants require different assumptions about the data

---

# 🆚 Naive Bayes vs Logistic Regression

| Feature               | Naive Bayes              | Logistic Regression |
| --------------------- | ------------------------ | ------------------- |
| Type                  | Probabilistic classifier | Linear classifier   |
| Training Speed        | Very Fast                | Fast                |
| High-Dimensional Data | Excellent                | Good                |
| Text Classification   | Excellent                | Excellent           |
| Feature Independence  | Assumed                  | Not required        |
| Output                | Class probabilities      | Class probabilities |
| Small Dataset         | Often strong             | Can work well       |

---

# 🧠 Key Takeaways

* Naive Bayes is a probabilistic classification algorithm.
* It is based on Bayes' Theorem.
* It assumes conditional independence between features given the class.
* `GaussianNB` is commonly used for continuous numerical features.
* `MultinomialNB` is widely used for count-based text classification.
* `BernoulliNB` works with binary features.
* `ComplementNB` can be useful for imbalanced text classification.
* Laplace smoothing helps prevent zero-probability problems.
* Naive Bayes is fast and effective for many classification tasks.
* It is especially popular in natural language processing.

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# 📚 Learning Resources

* [Introduction to Statistical Learning](https://www.statlearning.com/)
* [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [Andrew Ng Machine Learning Course](https://www.coursera.org/learn/machine-learning)
* [StatQuest](https://www.youtube.com/@statquest)

---

# 🚀 Future Improvements

* Add Gaussian Naive Bayes projects
* Add Spam Detection project
* Add Sentiment Analysis project
* Add Multinomial Naive Bayes text classification
* Add Bernoulli Naive Bayes experiments
* Add model comparison with Logistic Regression
* Add hyperparameter tuning experiments
* Add Naive Bayes implementation from scratch
* Add complete end-to-end classification projects

---

# 👨‍💻 Author

**Maganpreet Singh**

B.Tech Computer Science & Engineering

**Machine Learning | Data Science | Python**

---

# ⭐ Support

If you find this repository useful for learning **Naive Bayes Classifier**, consider giving it a ⭐ on GitHub.

> Keep learning. Keep building. Keep experimenting. 🚀
