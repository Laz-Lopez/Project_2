# Project_2






***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___
## <a name="project_description"></a>Project Description:
Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.

Create modules (acquire.py, prepare.py) that make your process repeateable and your report (notebook) easier to read and follow.

Ask exploratory questions of your data that will help you understand more about the attributes and drivers of customers churning. Answer questions through charts and statistical tests.

Construct a model to predict tax value of homes using regression techniques.

Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.


[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
a)Create deliverables:
- README
- final_report.ipynb
- working_report.ipynb
b) Build functional wrangle.py, explore.py, and model.py files
c) Acquire the data from the Code Up database via the wrangle.acquire functions
d) Prepare and split the data via the wrangle.prepare functions
e) Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.
f) Create a baseline model in predicting home cost and document the RSME.
g) Fit and train regression models to predict cost on the train dataset.
i) Evaluate the models by comparing the train and validation data.
j) Select the best model and evaluate it on the train data.
k) Develop and document all findings, takeaways, recommendations and next steps.


[[Back to top](#top)]

### Project Outline:
Acquire data
Prepare Data
Explore Data
Create Hypothesis
Test Model 
Conclusion

        
### Hypothesis
---
# Hypothesis 1

## $H_0$: Tax Value is independent of the Sqft of a home 

## $H_a$ : Tax Value is dependent of the Sqft of a home 

---

# Hypothesis 2

## $H_0$: Tax Value is independent of the Number of Bedrooms of a home has

## $H_a$ : Tax Value is dependent of Number of Bedrooms of a home has

---
# Hypothesis 3


## $H_0$: Tax Value is independent of the Number of Bathrooms a home has

## $H_a$ : Tax Value is dependent of Number of Bathrooms of a home has

---

### Target variable
number_services
monthly_avg
monthly_charges

### Need to haves (Deliverables):
acquire.py
prepare.py
Final_notebook.ipybn
this readme.md


### Nice to haves (With more time):
#### Functions to so no code is shown 



***

## <a name="findings"></a>Key Findings:
## There is more than one way to predict but simple is better and diving to deep will cause you to drown.

### Sqft plays a factor in value
### Number of Bathroom plays a factor in Value 
### Number of Bedrooms plays a factor in Value


[[Back to top](#top)]


***

## <a name="dictionary"></a>Data Dictionary  


### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|Area|The Sqft of the house|int64|
|Bathrooms| The Number of Bathrooms of the house |int64|
|Bedrooms| The Number of Bedrooms of the house |int64 |
|tax_value | The historical Tax Value of the house|int64|
|Sqft|the Sqft of the house broken in to 3 sizes, <1000,<2000,>2000| object|

***
[[Back to top](#top)]
## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Wrangle steps: 
Try to Make pretty pictures
Repeat until you get something you understand.

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:

    - explore.py



### Takeaways from exploration:


***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]


### Stats Test 1: Chi2:

#### Hypothesis 1:
# $H_0$: Tax Value is independent of the Sqft of a home 

## $H_a$ : Tax Value is dependent of the Sqft of a home 

#### Alpha value:

- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
We reject the null hypothesis.

#### Summary:
While I still do not fully grasp this process it was completed
***
### Stats Test 2: Chi2


#### Hypothesis 2:
## $H_0$: Tax Value is independent of the Number of Bedrooms of a home has

## $H_a$ : Tax Value is dependent of Number of Bedrooms of a home has
#### Alpha value:
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
We reject the null hypothesis.


#### Summary:
While I still do not fully grasp this process it was completed

### Stats Test 3: Chi2

#### Hypothesis 3:
# $H_0$: Tax Value is independent of the Number of Bathrooms a home has

## $H_a$ : Tax Value is dependent of Number of Bathrooms of a home has

#### Alpha value:
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
We reject the null hypothesis.


#### Summary:
While I still do not fully grasp this process it was completed


## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Model Preparation:

### Baseline
    
- Baseline Results: 
Our baseline accuracy in all cases on the Dataset is :

RMSE Mean:
248150.10218076012
----------------
RMSE Median:
250857.8604903843

- Selected features to input into models:
    - features = Area, Bathrooms, Bedrooms

***

### Models and R<sup>2</sup> Values:
- Will run the following regression models:

    

- Other indicators of model performance with breif defiition and why it's important:

    


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

RMSE for Lasso & Lars
Training/In-Sample:  219234.24 
Validation/Out-of-Sample:  217879.84
R2: 0.22
_______________________________________________
RMSE for OLS using LinearRegression
Training/In-Sample:  219233.87 
Validation/Out-of-Sample:  217881.69
R2: 0.22
_______________________________________________
RMSE for Polynomial Model, degrees=2
Training/In-Sample:  219176.92 
Validation/Out-of-Sample:  217894.86
R2: 0.22





***

## <a name="conclusion"></a>Conclusion:
Sqft plays a factor in value
Number of Bathroom plays a factor in Value
Number of Bedrooms plays a factor in Value

### Steps to Reproduce
Your readme should include useful and adequate instructions for reproducing your analysis and final report.

For example:

1)You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the titanic_db.passengers table. Store that env file locally in the repository.

2)clone my repo (including the wrangle.py, explore.py, and model.py) (confirm .gitignore is hiding your env.py file)

3)libraries used are pandas, matplotlib, seaborn, numpy, sklearn,scipy, math.

4)you should be able to run churn_report.


[[Back to top](#top)]
