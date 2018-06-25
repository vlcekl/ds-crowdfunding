# Kickstarter data - inferential statistics

The Kickstarter dataset can provide useful statistical information about the relations between different features 
and for making predictions about the eventual success of the proposal.
Here we will analyze the significance of different variables as predictors of target variables and correlations between the predictor variables
for possible dimensionality reduction. The results should provide us with a
starting point for creating predictive models useful both for investors and proposers.

## 
The primary target variables useful for both groups (investros and proposers), are (i) the binary catergorical variable 'state'
representing the success or failure of the funding campaign, and (ii) the (roughly) continuum variable 'pledged' representing
the amount of money ultimately pledged by the investors.

The primary predictor variable whose significance will be tested are 'goal', representing the target funding amount set by the proposers.
Onother predictor variables include 'category' in which the project falss, and the country of 'origin' of the proposers.

A special variable is 'staff pick', which can be useful for investors, but is not available as a predictor for proposers. In fact, if 'staff pick' is found to have positive influence on funding, the proposers can optimize their proposals to increaset the chance of being picked. In this case 'staff pick' can be considered as binary target variable.

We noted in the exploratory data analysis, that there appear to be significant
correlations between 'catogory' and country of 'origin' variables, and also
between 'category' and 'goal' variables. Significance of these correlations
will be tested.
