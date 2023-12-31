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
![data](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week%20_1_Projects/data.png)
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
```python code
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
data = np.loadtxt('Sample.data.csv', delimiter=',')

# Split the data into features and target variable
X = data[:, :-1]
y = data[:, -1]

# Split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
```

---

### 5.	Model Selection: 
- In this step I selected a suitable machine learning algorithm  for the model which is random forests.
Random forest is an ensemble technique that works on decision trees. It uses the bootstrap aggregation (bagging)
 technique over multiple decision trees. As the name bootstrap suggests, the decision tree is trained on several 
 data sets drawn from the original data set, with replacement (reusing the same data samples multiple times). Multiple 
such decision trees are trained, and the final outcome is based on the average of outcomes of individual trees.
- Random Forest algorithm is far more accurate at predictive analytics in general. It is one of the best algorithms 
used for regression and classification analysis
```python code
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = np.loadtxt('Sample.data.csv', delimiter=',')

# Split the data into features and target variable
X = data[:, :-1]
y = data[:, -1]

# Split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Create the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = clf.predict(X_val)
accuracy_val = np.mean(y_pred_val == y_val)

# Print the validation accuracy
print('Validation accuracy:', accuracy_val)

# Evaluate the model on the test set
y_pred_test = clf.predict(X_test)
accuracy_test = np.mean(y_pred_test == y_test)

# Print the test accuracy
print('Test accuracy:', accuracy_test)

# Save the model
clf.save('churn_model.pkl')
```
---

### 6.Model Training:
- Here, I  trained the model on the training data using Random forest algorithm.
 Once I had the chosen algorithm, I  needed to train the model on the prepared data. 
 This involves feeding the algorithm the data and allowing it to learn the relationships between 
 the different variables. Tune hyperparameters using techniques like cross-validation to optimize model performance.
 ```python code
 import numpy as np
from sklearn.ensemble import RandomForestClassifier
data = np.loadtxt('Sample.data.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Create the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)
```
---

### 7.Model Evaluation:
- After training the model, I evaluated the model's performance on the validation and test sets using relevant 
evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. This provided an idea of the model's
 potential performance on fresh data. In order to make sure the model makes accurate predictions; I will compare its 
 performance to that of a baseline model (such as a random guess).
 ```python code
 import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
data = np.loadtxt('Sample.data.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = clf.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)
roc_auc_val = roc_auc_score(y_val, y_pred_val)

# Print the validation metrics
print('Validation accuracy:', accuracy_val)
print('Validation precision:', precision_val)
print('Validation recall:', recall_val)
print('Validation F1-score:', f1_val)
print('Validation ROC-AUC:', roc_auc_val)

# Evaluate the model on the test set
y_pred_test = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_pred_test)

# Print the test metrics
print('Test accuracy:', accuracy_test)
print('Test precision:', precision_test)
print('Test recall:', recall_test)
print('Test F1-score:', f1_test)
print('Test ROC-AUC:', roc_auc_test)
```
---

### 8.Deployment:
- Once the model is trained and validated and I felt satisfied that the model is able
 to predict customers behavior and churn, I went on to deploy it into the production environment, 
 where it can be used to make real-time predictions on new customer data.

```python code
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the model
model = pickle.load(open('churn_model.pkl', 'rb'))

# Make a prediction on a new customer
new_customer = np.array([30, 'M', 'New York', 100000, 100, 1000, 1])

# Predict the churn probability for the new customer
churn_probability = model.predict_proba(new_customer)[0][1]

# If the churn probability is greater than a certain threshold, then predict that the customer will churn
if churn_probability > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is unlikely to churn.')
```
--- 

### 9.Monitoring and Maintenance:
- Once 	I have deployed the model in the production environment,
 I will continuously monitor the model's performance and retrain it periodically with 
 new data to ensure its accuracy remains high. This will involve checking for issues, 
 modifications as needed, and prospective issues. The purpose of maintenance and monitoring
  is to guarantee the ongoing use, dependability, and security of the model, by performing software
   updates, hardware maintenance, configuring management.

--- 

### 10.Feedback Loop:
- A feedback loop is an ongoing, iterative process for fine-tuning and enhancing the
 model based on user feedback and assessments of its effectiveness.
  The model will rely on the feedback loop process since it will enable me to 
  continuously enhance and refine the model. It makes sure the model is in line with the 
  facts and specifications of the real world and is flexible enough to change with the times. 
  Feedback loops also aid in spotting and fixing any problems like bias or overfitting that mightnot be obvious in the early phases of development.
---