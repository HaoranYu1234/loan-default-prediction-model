# A Machine Learning Perspective on Predicting Loan Default Risks in Financial Services

## Research question: How can loan default prediction models be developed and deployed using machine learning techniques?

## Research participant👨‍🔬👩‍🔬🥼:
- Haoran(Alex) Yu

## Research timeline ⌚⏳⏲:
General timeline: 2024 Augest - 2024 December

## Research proposal and planning

### Research motivation 🧐💰💵:
Understanding and predicting loan defaults is crucial for financial institutions to manage risk effectively. Accurate predictions can help in making informed lending decisions, reducing the occurrence of bad debts, and ensuring financial stability. Also by performing machine learning, lenders can refine their criteria for approving loans, ensuring that they are lending to individuals and businesses with a lower risk of default. This can lead to more responsible lending practices and a healthier credit market.

### Data set used for the research:
[https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1?select=Loan_status_2007-2020Q3.gzip)

### Research methodology 📏🧰🦾:
- Data Preprocessing: 
  - Data cleaning
  - Create new features and convert categorical variables
  - Scaling and normalization
- Exploratory data analysis：
  - Understanding the distribution and relationship between predictors and response vasriables
- Model training and evaluation
- Model optimization
- Documentation and reporting

### Potential and pending model selection 🛠⛏⚒:
- Logistic regression
- Linear regression
- Decision trees
- Random Forest
- Gradient boosting machines (e.g., XGBoost, LightGBM)
- Support vector machines (SVM)

### Future steps 🔮📈📉
After the completion of this current project, a API can be created using this trained model. At that time, a new project will be created.

## General Concepts of Supervised Learning

Supervised learning involves training models to learn the mapping between input variables (predictors) and output variables (responses) based on labeled datasets. The primary goals of supervised learning are:

1. **Prediction**: To accurately predict outcomes for new, unseen inputs.
2. **Estimation**: To understand how predictors influence the response.
3. **Model Selection**: To determine the best model for making predictions.
4. **Inference**: To evaluate the reliability of predictions and the understanding of data relationships.

### Key Components of Supervised Learning

- **Outcome Measurement (Y)**: Known as the dependent variable or response, which the model aims to predict.
- **Predictor Measurements (X)**: The independent variables or features used to predict the outcome.

The relationship between inputs \( X \) and output \( Y \) is often modeled as:

$$
Y = f(X) + \epsilon
$$

where \( f(X) \) is the true underlying relationship and \( \epsilon \) represents the error term.

The objective is to learn an approximation \( g(X) \) of \( f(X) \) from the training data \( D_{\text{train}} \), consisting of observations \( (x_1, y_1), \ldots, (x_n, y_n) \). The predictor \( g(X) \) should generalize well to new test data \( x^* \).

### Parametric and Non-Parametric Methods

- **Parametric Methods**: Assume a specific functional form for \( f(X) \), such as linear models.
- **Non-Parametric Methods**: Do not make assumptions about the form of \( f(X) \) and are more flexible but often require more data.

### Model Selection Process

Model selection involves comparing different types of models, such as parametric vs. non-parametric, using systematic criteria to find the one that balances fit and complexity.

### Mathematical Framework

The process of learning \( f \) can be viewed as an optimization problem, where the aim is to minimize the expected loss:

$$
\mathbb{E}[(Y - g(X))^2]
$$

In practice, this is often approximated using the training dataset:

$$
\text{Minimize: } \frac{1}{n} \sum_{i=1}^n (y_i - g(x_i))^2
$$

This provides the foundation for model fitting and evaluation in supervised learning.
