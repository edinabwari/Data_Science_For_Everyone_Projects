## **Week One project: Churn Prediction for Sprint.**

Question

Imagine you're working with Sprint. They're really keen on figuring out how many customers might decide 
to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have 
left before, as well as info about who these customers are, what they've bought, and other things like that.
 So, if you were in charge of predicting customer churn how would you go about using machine learning to make
 a good guess about which customers might leave? Like, what steps would you take to create a machine learning 
 model that can predict if someone's going to leave or not?
---

## **Answer**
- The customer churn prediction (CCP) is one of the challenging problems in bussiness today. This is a term that refers
to the number of customers who terminates the services of a company or  switches to  another. It  really  decreases the  profit
 margin in a company if it loses its customers. Predicting the number of customers wo will leave the company early can be a
 large revenue of source. With this machine learning model, Sprint can take measures to retain  profitable customers and
 work on areas where customer service is lacking.
 
- There are various processes involved in building a machine learning model to forecast client churn. 
Here is a general outline of how I did this work.
---
### 1. Data	Collection and preparation.
---
- I prepared the bunch of past data and gather information available about the consumers by using a variety of historical 
data about when previous  customers left as well as information about who these customers are, their demographics,
 purchase history, usage patterns, customer service interactions, what they've purchased, and other similar information.
 The data is in the ***Sample.data.csv file***
 
 ```Python code
 import pandas as pd
from google.colab import files
uploaded = files.upload()
import io
data = pd.read_csv(io.BytesIO(uploaded['Sample.data.csv']))
data.sample(10)
```

- I then proceeded in cleaning up and preprocess the data, this  involved dealing with difficulties in data formatting,
 cheking for outliers that I found none, and  checking for missing values in the data.
 The ***Sample.data.csv*** data needed to be cleansed and preprocessed so that the model can make  good use of it.


```Python code to check for missing values.
import pandas as pd
# Load the data from the CSV file and Checked for missing values in each column
data = pd.read_csv('Sample.data.csv')
missing_values = data.isnull().sum()
print("Missing Values Count per Column:")
print(missing_values)
```

```python code for cheking outliers
import pandas as pd
import numpy as np
from scipy import stats
# Load the data from the CSV file
data = pd.read_csv('Sample.data.csv')
# Define a threshold for the Z-score
z_score_threshold = 3  # Adjust this threshold as needed
# Calculate the Z-scores for numerical columns (exclude non-numeric columns)
numeric_cols = data.select_dtypes(include=[np.number])
z_scores = np.abs(stats.zscore(numeric_cols))
# Identify outliers based on the Z-score threshold
outliers = (z_scores > z_score_threshold).any(axis=1)
# Print the rows with outliers
print("Rows with Outliers:")
print(data[outliers])
```
---

### 2.EDA, (exploratory data analysis).
- In this step, the cleaned data that I have had to be analysed in order to find patterns,
 uncover anomalies, test hypotheses, and double-check assumptions using summary statistics 
 and graphical representations.
- I came up with a correlation matrix and a histogramand to visualize the data as a heatmap and understand the relationships
  between numerical features and to identify patterns and correlations. I then explored the 
  to gain insights into the relationships between different features and churn. 
  I needed to analyze trends in the data to find the main reasons 
 behind customer churn. And visualize the data. 
- Bellow is the code that I used.
```python code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Histograms from the data
data.hist(bins=20, figsize=(12, 10))
plt.suptitle("Histograms", y=1.02)
plt.show()
```
![Correlation_matrix](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week%20_1_Projects/Correlation_matrix.png)
![Histogram](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week%20_1_Projects/Histogram.png)
- from the  the correlation diagram from the **Sample.data.cvs** file I could see that there are a number of 
factors that can influence a customer's company contract length, including their age, income, and usage patterns.
With this information I could tell which customers are most likely to cancel their contracts and to target marketing
and retention strategies at these customers.
- for instance, looking at the correlation diagram in Age and contract length: There is a strong negative correlation 
between age and contract length, meaning that older customers are more likely to have shorter contracts. This may be
 because older customers are more likely to be retired or have other commitments. However, in Income and contract length: 
 There is a strong positive correlation between income and contract length, meaning that 
customers with higher incomes are more likely to have longer contracts because they are likely to be able to afford a long-term contract 
with the company and may also be more likely to value the benefits that come with a longer contract, 
---

### 3.Feature Engineering:
- In this step I focused on choosing, modifying, and converting raw data into features that may be applied in supervised learning. 
 I choose to calculate the average number of calls made or texts sent per day from the customers data.
 ```python codeimport pandas as pd

data = pd.read_csv('Sample.data.csv')
#Feature Engineering: Calculate average calls made per day

data['average_calls_per_day'] = data['calls_made'] / data['contract_length']

# I calculated the average texts sent per day:

data['average_texts_per_day'] = data['texts_sent'] / data['contract_length']

data.to_csv('Feature_Engineered_Sample.data.csv', index=False)

print(data)
```
---

### 4.Data Splitting:
- I will proceed by splitting the dataset into three categories. This includes training, validation, and test sets.
 The training set I used to train the model, the validation set I used to tune hyperparameters,
  and the test set I  used to evaluate the model's performance.

---

### 5.	Model Selection: 
- A suitable machine learning algorithm will then be selected for the model. Logistic regression, 
decision trees, random forests, support vector machines, and neural networks are a few of the often-used options.
 I'll also test out many algorithms to see which one works best for my particular dataset before carefully choosing the top algorithm.

---

### 6.Model Training:
- Here, I will train the model on the training data using the chosen algorithm. Once I have the chosen algorithm, I would need to train the model on the prepared data. This involves feeding the algorithm the data and allowing it to learn the relationships between the different variables. Tune hyperparameters using techniques like cross-validation to optimize model performance.
---

### 7.Model Evaluation:
- After training the model, I will proceed to evaluate the model's performance on the validation and test sets using relevant evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. This would provide an idea of the model's potential performance on fresh data. In order to make sure the model makes accurate predictions; I will compare its performance to that of a baseline model (such as a random guess).
---

### 8.Deployment:
- Once the model is trained and validated and I feel satisfied that the model is able to predict customers behavior and churn, I will go on to deploy it into the production environment, where it can be used to make real-time predictions on new customer data.

--- 

### 9.Monitoring and Maintenance:
- Once 	I have deployed the model in the production environment, I will continuously monitor the model's performance and retrain it periodically with new data to ensure its accuracy remains high. This will involve checking for issues, modifications as needed, and prospective issues. The purpose of maintenance and monitoring is to guarantee the ongoing use, dependability, and security of the model, by performing software updates, hardware maintenance, configuring management.

--- 

### 10.Feedback Loop:
- A feedback loop is an ongoing, iterative process for fine-tuning and enhancing the model based on user feedback and assessments of its effectiveness. The model will rely on the feedback loop process since it will enable me to continuously enhance and refine the model. It makes sure the model is in line with the facts and specifications of the real world and is flexible enough to change with the times. Feedback loops also aid in spotting and fixing any problems like bias or overfitting that might not be obvious in the early phases of development.
---