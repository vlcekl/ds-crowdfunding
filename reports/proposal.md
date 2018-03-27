# Success Indicators of Crowdfunding Projects
### _Lukas Vlcek_

---

## Problem description
What are crowdfunding platforms and why people go this route?
When deciding
  + Kickstarter and Indiegogo
  + What are the characteristics of funded projects?
  + Can we predict a successful delivery?
  + Predicting delays.
  + Explain reasons for the outcomes.
What are the differences between the two platforms in terms of types of the projects, startups, funding success, etc.

## Who cares?
The analysis of project will help to make decison whether to invest the time and effort 
Startups: Clearly presented statistics of the current projects and the overall success rate should also inform prospective proposee , how crowded and competitive the . and therefore.
Identify promising areas and avoid those with lower chance of success.
Similar projects  will help them better decisions whether to invest their time and effort into the project.
Investors considering funding of a particular project should be able to make a better informed decision about the chances of their investment paying off. 
People that want to know, whether it is even worth to start thinking about ideas that could be crowdfunded.


## Datasets
The primary source of data will be obtained from online, monthly updated repositories of crowdfoonding projects collected by a
web scraping company [Web Robots](https://webrobots.io/) for Kickstarter and Indiegogo platforms.
The data is available in JSON and CSV formats.

  1. [Kickstarter data (4/2014 - present): https://webrobots.io/kickstarter-datasets/](https://webrobots.io/kickstarter-datasets/)
  2. [Indiegogo data (5/2016 - present): https://webrobots.io/indiegogo-dataset/](https://webrobots.io/indiegogo-dataset/)

(Information about the project success? Did they deliver?)

## Approach

To help startups and investors make their decisons, I will try to predict the probabilities of two categorical (binary) variables:

 1. a given proposal will achieve its funding goals
 2. a funded proposal will succeed in achieving its goals

Depending on the quality and quantity of the data, I may consider refinement
of these categories into the percentage of funding, or dividing project
success into major (on time) and minor (delivered but late).

The main approach will be based on supervised learning in the form of a logistic regression.

I will use older records as training data and newer records as testing data.

If time allows, I may consider exploring usefulness of unsupervised ML
approaches (DL?) to find correlations between different categories of input
data.

Dimensionality reduction?

Is the length of the description or some keywords related to successful funding campaign?
Maybe effects, such as TL;DR; or 

## Deliverables
  + What are the characteristics of funded projects?
  + Can we predict a successful delivery?
  + Predicting delays.
  + Explain reasons for the outcomes.

Project files deployed to github.

Because of its nature, it is likely that whole crowds of people would be interested in this subject,
It may be suitable to present the results in a blog post. This presumes that interesting insight are gained from the data.

