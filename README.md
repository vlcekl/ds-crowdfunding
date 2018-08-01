ds-crowdfunding
==============================

## Success indicators of crowdfunding projects

A data science project aiming to predict success or failure of Kickstarter crowdfunding
projects based on historical data.

Cleaned data for ~200,000 projects are located in data/processed directory

Data wrangling, exploration, statistical analysis, and predictive modeling are contained in
corresponding Jupyter notebooks in 'notebooks' directory.

The Python-based project requires pandas and scikit-learn libraries.

The modeling pipeline processes multiple features:
goal amout, category type, country of origin, project name, and project
description. The supervised classification is accomplished using logistic
regression, for which categorical data are converted by one-hot encoding, and
textual data by count vectorizer.

The predictive abilities match or exceed human assessment of the project
(staff pick).
