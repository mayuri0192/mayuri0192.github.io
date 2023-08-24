---
layout: post
title: Unveiling Two-Class Classification Algorithms: A Comparative Study
description: Understanding Two-Class Classification Algorithms with respect to Dataset type, Accuracy, Training time, Linearity, Number of parameters, Number of features and Computation.
date: 2023-08-24
tags: machine learning algorithms, interview preperation, job hunting, publication
comments_id: 1
---

Lets consider a dataset to understand two class classification.

This dataset involves classifying whether a transaction is fraudulent or not based on multiple features related to the transaction details.

| Transaction_ID | Amount | Merchant  | Location      | Time   | Fraudulent |
|---------------|--------|-----------|---------------|--------|------------|
| 1             | 200    | Amazon    | New York      | 10:00  | No         |
| 2             | 500    | Walmart   | Los Angeles   | 15:30  | No         |
| 3             | 50     | Starbucks | Seattle       | 12:15  | No         |
| 4             | 150    | Target    | Chicago       | 18:00  | No         |
| 5             | 800    | Amazon    | New York      | 09:30  | Yes        |
| 6             | 300    | Walmart   | Los Angeles   | 14:20  | No         |
| 7             | 500    | Starbucks | Seattle       | 16:30  | No         |
| 8             | 1200   | Amazon    | New York      | 23:45  | Yes        |
| 9             | 90     | Target    | Chicago       | 08:10  | No         |
| 10            | 100    | Walmart   | Los Angeles   | 20:00  | No         |


In this dataset, each row represents a transaction with various features such as "Amount" (transaction amount), "Merchant" (merchant name), "Location" (transaction location), "Time" (transaction time), and "Fraudulent" (whether the transaction is fraudulent or not).

# Understanding patterns in dataset:

## Linearity: 
Imagine you're plotting the features of a transaction against the likelihood of it being fraudulent. If a straight line can be drawn through the scatter plot that accurately separates the fraudulent transactions from the legitimate ones, we have a linear relationship. This suggests that as the values of certain features change, the probability of fraud also changes proportionally.
Lots of machine learning algorithms make use of linearity. In Azure Machine Learning designer, they include:

- logistic regression
- Support vector machines

1. Logistic Regression: This algorithm assumes a linear relationship between the input features and the log-odds of the target variable. If the data exhibits linear separability, logistic regression can draw a boundary that effectively distinguishes between the two classes, allowing for accurate classification of fraudulent and legitimate transactions.

2. Linear Support Vector Machines (SVM): Similar to logistic regression, linear SVMs rely on the existence of linearly separable data. They seek a straight-line boundary that maximizes the margin between the two classes. If the fraud patterns are linear, SVM can efficiently detect anomalies.
For Data with a nonlinear trend: Using a logistic regression or SVM methods would generate much larger errors than necessary.

[<img align="center" src="/assets/linear.PNG" width="200"/>](/assets/linear.PNG)

## Non-Linearity and Complex Patterns:


| Transaction_ID | Amount | Merchant  | Location   | Time   | Fraudulent |
|---------------|--------|-----------|------------|--------|------------|
| 1             | 200    | Amazon    | New York   | 10:00  | No         |
| 2             | 500    | Walmart   | Los Angeles| 15:30  | No         |
| 3             | 1000   | Unknown   | London     | 20:45  | Yes        |
| 4             | 150    | Starbucks | Seattle    | 12:15  | No         |
| 5             | 800    | Unknown   | New York   | 09:30  | Yes        |
| 6             | 50     | Target    | Chicago    | 18:00  | No         |
| 7             | 300    | Unknown   | Paris      | 14:20  | Yes        |
| 8             | 700    | Walmart   | Los Angeles| 23:45  | No         |
| 9             | 1200   | Unknown   | Tokyo      | 08:10  | Yes        |
| 10            | 90     | Starbucks | Seattle    | 16:30  | No         |


In this dataset, non-linearity and complex patterns can arise when certain transactions show seemingly unrelated combinations of features that indicate fraud. For instance:

Transaction 3: A high transaction amount combined with an unusual merchant location (London) and an atypical time (20:45) suggests fraud. This complex pattern might be missed by linear models.

Transaction 5: Despite the high transaction amount, the combination of "Unknown" merchant and location (New York) along with an early time (09:30) indicates potential fraud. Non-linearity is needed to capture this intricate relationship.

Transaction 9: A large transaction amount paired with an unusual location (Tokyo) and an early time (08:10) is a complex pattern that points towards fraud.

[<img align="center" src="/assets/non-linear.PNG" width="200"/>](/assets/non-linear.PNG)
## Applying Non-Linear Models:

To detect fraud patterns like these, non-linear classification algorithms like decision trees, random forests, or support vector machines with non-linear kernels would be more effective. These algorithms can uncover complex relationships among features that might not be captured by linear models.

In the Fraud Detection domain, where fraudsters continuously evolve their tactics, non-linearity and complex patterns are common. By using algorithms that can decipher these intricate connections, data scientists can enhance fraud detection accuracy and mitigate financial risks.


| Description         | Few Features                                                | Many Features                                                                                               |
|--------------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Model Complexity              | With a small number of features, models tend to be simpler and more interpretable. Linear models like Logistic Regression and Linear SVM might work well when there are only a few crucial features that contribute to fraud detection. | As the number of features increases, models can become more complex. This complexity can capture intricate relationships in the data but might also result in overfitting if not managed properly. |
| Overfitting and Generalization | When the number of features is limited, models are less likely to overfit the training data. They may generalize better to new, unseen data. | With many features, there's a higher risk of overfitting, where the model memorizes the training data but struggles to perform well on new data. Regularization techniques become more important to prevent overfitting. |
| Feature Importance       | In simpler models, it's easier to identify which features have a significant impact on fraud detection. Interpretability is higher when fewer features are present. | In more complex models with many features, identifying which ones are truly important becomes challenging. Feature selection or extraction techniques might be needed to manage this complexity. |
| Computational Demands    | Models with fewer features train faster and require less computational power. This is beneficial when quick model iteration is necessary. | Models with a large number of features demand more computational resources and time for training. This can slow down the experimentation process. |
|Curse of Dimensionality | -- | When the number of features is much larger than the number of observations, the dataset is considered high-dimensional. High dimensionality can lead to challenges in finding meaningful patterns, increased risk of overfitting, and increased data requirements.| 




# Model Complexity: 
The number of features in a dataset can significantly impact the application of classification models, especially in scenarios like fraud detection. Let's explore how the number of features affects the process of applying classification models using the example of fraud detection:

## Few Features:

**Logistic Regression**: This algorithm works well when you have a small number of features. It's interpretable, efficient to train, and can handle linear relationships between features and the target variable.

**Naive Bayes**: Naive Bayes is effective for small feature sets. It's particularly useful when dealing with text or categorical data and assumes independence between features.

**Decision Trees**: Decision trees are intuitive and can handle few features effectively. They can capture non-linear relationships in the data and provide insights into feature importance.

## Many Features:

**Random Forest**: Random forests are an ensemble of decision trees and can handle a larger number of features. They reduce overfitting and provide robust predictions by aggregating the output of multiple trees.

**Gradient Boosting**: Gradient Boosting algorithms like XGBoost and LightGBM are powerful for handling complex datasets with many features. They sequentially build a strong model by focusing on misclassified instances, making them effective for high-dimensional data.

**Support Vector Machines (SVM)**: SVMs can handle many features by mapping the data into a higher-dimensional space. Using appropriate kernels, SVMs can capture complex relationships even in high-dimensional data.

**Neural Networks**: Deep learning models, such as neural networks, are known for their ability to handle high-dimensional and complex data. They can automatically learn intricate patterns from the data, but they might require larger amounts of data for training.

## Overfitting and Generalization:
When it comes to handling overfitting and ensuring good generalization, you'll want to choose classification algorithms that offer ways to mitigate overfitting and promote better generalization on new, unseen data. Here are some suitable classification algorithms for addressing overfitting and achieving better generalization:
**Few Features**: When the number of features is limited, models are less likely to overfit the training data. They may generalize better to new, unseen data.
**Many Features**: With many features, there's a higher risk of overfitting, where the model memorizes the training data but struggles to perform well on new data. Regularization techniques become more important to prevent overfitting.
**Regularized Logistic Regression**: Regularization techniques like L1 (Lasso) and L2 (Ridge) regularization can help prevent overfitting by adding penalty terms to the model's loss function. These penalties discourage overly complex models, leading to better generalization.

**Regularized Linear Support Vector Machines (SVM)**: Regularization in SVMs can help control overfitting. By tuning the regularization parameter (C), you can adjust the trade-off between maximizing the margin and minimizing the classification error.

**Random Forest with Max Depth Constraint**: Random forests can tend to overfit if allowed to grow deeply. Limiting the maximum depth of individual decision trees can prevent overfitting and improve generalization.

**Gradient Boosting with Early Stopping**: Gradient boosting algorithms often use boosting iterations, which can lead to overfitting. Implement early stopping by monitoring performance on a validation set. Stop training when performance plateaus to avoid overfitting.

**Neural Networks with Regularization**: In deep learning, techniques like dropout (randomly disabling neurons during training) and L2 regularization can help prevent overfitting in neural networks. Additionally, using smaller network architectures and optimizing hyperparameters can improve generalization.

**SVM with Kernel Tricks**: When using SVMs, choosing appropriate kernel functions (such as the radial basis function) can help capture complex relationships without overfitting. Cross-validation can assist in tuning the kernel parameters.

**Ensemble Method**s: Bagging and boosting ensemble methods (like AdaBoost) combine multiple models to improve generalization and reduce overfitting by focusing on correctly classifying difficult instances.

## Feature Importance:

**Few Features**: In simpler models, it's easier to identify which features have a significant impact on fraud detection. Interpretability is higher when fewer features are present.
**Many Features**: In more complex models with many features, identifying which ones are truly important becomes challenging. Feature selection or extraction techniques might be needed to manage this complexity.

**Decision Trees and Random Forests**: Decision trees and random forests provide a natural way to assess feature importance. Features that split data into pure classes (resulting in pure nodes) are considered more important. Random forests aggregate feature importance from multiple trees, offering a robust measure of feature relevance.

**Gradient Boosting Algorithms (XGBoost, LightGBM)**: Gradient boosting algorithms offer feature importance scores based on how often a feature is used across multiple boosting iterations. This provides a measure of a feature's contribution to improving model performance.

**Linear Models with Coefficients**: Linear models like Logistic Regression and Linear SVM directly provide coefficients for each feature. Larger absolute coefficients indicate higher importance. Regularized linear models can help control feature importance by preventing overemphasis on specific features.

**Permutation Feature Importance**: While not tied to a specific algorithm, permutation feature importance is a technique that works with any classifier. It involves shuffling a feature's values and measuring the impact on model performance. The larger the drop in performance, the more important the feature.

**Tree-Based Models (Decision Trees, Random Forests) with Feature Importance Visualization**: Some libraries offer visualization tools to display feature importance within tree-based models. These visualizations can help interpret how features affect the model's decisions.

**Recursive Feature Elimination (RFE)**: This method works with any classifier and recursively eliminates less important features while measuring the impact on model performance. The remaining features are considered more important.

**LASSO (Least Absolute Shrinkage and Selection Operator)**: LASSO regularization can drive the coefficients of less important features to zero, effectively excluding them from the model and highlighting important features.

## Curse of Dimensionality

The *"Curse of Dimensionality"* refers to the challenges that arise when dealing with high-dimensional data, where the number of features greatly surpasses the number of observations. In the context of fraud detection, this challenge can impact the performance of classification algorithms and hinder their ability to accurately distinguish between genuine and fraudulent transactions.

**Sparse Data**: As the number of dimensions (features) increases, the data points become more sparse. In fraud detection, this means that instances of fraud become rare events scattered across the high-dimensional space. Detecting patterns and anomalies in such sparse data becomes challenging.

**Increased Complexity**: High-dimensional data often contains complex relationships that are difficult to visualize and comprehend. Fraud detection algorithms might struggle to identify relevant features or distinguish between legitimate and fraudulent transactions.

### Algorithms to Handle the Curse of Dimensionality:

**Feature Selection and Extraction**: To combat dimensionality, algorithms like Recursive Feature Elimination (RFE) and Principal Component Analysis (PCA) can be applied. RFE iteratively eliminates less important features, reducing dimensionality while retaining essential information. PCA transforms the data into a lower-dimensional space, capturing most of the variance.

**Ensemble Methods**: Ensemble algorithms like Random Forest and Gradient Boosting can handle high-dimensional data more effectively. They create multiple models that collectively capture complex relationships while minimizing overfitting.

**Regularization Techniques**: Algorithms like L1 regularization (Lasso) encourage sparsity by driving certain coefficients to zero. This helps to select the most relevant features and mitigate the curse of dimensionality.

**Dimensionality Reductio**n: Techniques like t-SNE (t-Distributed Stochastic Neighbor Embedding) and UMAP (Uniform Manifold Approximation and Projection) can reduce high-dimensional data into lower-dimensional representations that retain essential structure.

**Neighborhood-Based Methods**: Algorithms like K-Nearest Neighbors (KNN) and Local Outlier Factor (LOF) consider local structures rather than global patterns. They can be effective in detecting anomalies within high-dimensional data.

**Domain Knowledge and Preprocessing**: Utilizing domain knowledge to carefully select features or engineer new ones can mitigate the curse of dimensionality. Preprocessing techniques like scaling and normalization are also crucial to ensure that algorithms perform optimally.

Example of links: [here](https://github.com/mayuri0192)!

