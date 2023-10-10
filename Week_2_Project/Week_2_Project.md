## **Week Two project:  Comparison of a model to predict booking prices on Airbnb.**
---

** Question: **
Let’s say we want to build a model to predict booking prices on Airbnb. Between 
linear regression and random forest regression, which model would perform better and why?
---

## **Answer**
- I will compare both the two algorithm and later on come up with a conclusion of which algorithm I will use and my reasons.
I will fist of all start with Linear regression and later on focus on Random forest regression.
---
## **Linear regression**
- This is a statistical method that is used for predictive analysis of a variable based on the value of another variable.
 It makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, for this case 
 to predict the price on Airbnb I will use numeric variables like price of a room, the number of rooms, guest capacity 
 in a room. The variable I  want to predict is called the dependent variable while the variable I am  using to predict 
 the other variable's value is called the independent variable. Linear regression algorithm shows a linear relationship 
 between a dependent (y) and one or more independent (y) variables.

 - Here is a look at the data I used to 
 ```python
 import pandas as pd
from google.colab import files
uploaded = files.upload()
import io
data = pd.read_csv(io.BytesIO(uploaded['Airbnb.data.csv']))
data.sample(10)
```
![data](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week_2_Project/data.png)