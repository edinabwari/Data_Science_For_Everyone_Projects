## **Week Two project:  Comparison of a model to predict booking prices on Airbnb.**
---

Question:

- Let’s say we want to build a model to predict booking prices on Airbnb. Between 
linear regression and random forest regression, which model would perform better and why?
---

## **Answer**
- I will compare both the two algorithm and later on come up with a conclusion of which algorithm I will use and my reasons.
I will  start with Linear regression and later on focus on Random forest regression.
- Some of the key factors that affect the price of Airbnb are discussed here [pdf](https://pdfs.semanticscholar.org/64d2/77ee8949d2eb5e5e14929d15ea008cb5b836.pdf)
---
## **Linear regression**
- This is a statistical method that is used for predictive analysis of a variable based on the value of another variable.
 It makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, for this case 
 to predict the price on Airbnb I will use numeric variables like price of a room, the distance, guest capacity 
 in a room, review score rating.

 - Here is a look at the data I used. I generated the data from [mockaroo](https://www.mockaroo.com/)
 ```python
 import pandas as pd
from google.colab import files
uploaded = files.upload()
import io
data = pd.read_csv(io.BytesIO(uploaded['Airbnb.data.csv']))
data.sample(10)
```
![data](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week_2_Project/data.png)

- The variable I  want to predict is called the dependent variable while the variable I am  using to predict 
 the other variable's value is called the independent variable. In this case, the _dependent variable_  is the _Price_ of the Airbnb room is the dependent variable
This is the variable I want to estimate based on other factors like _distance_, _review_, etc.

- I compared compared the price of the Airbnb rooms with their review score rating using this code.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Airbnb.data.csv')

# Get the review_score_rating and price attributes
review_score_rating = df['review_score_rating']
price = df['price']

# Create a linear regression model
model = np.polyfit(review_score_rating, price, 1)

# Calculate the predicted price for each review score rating
predicted_price = np.polyval(model, review_score_rating)

# Create a scatter plot of the price and review score rating attributes
plt.scatter(review_score_rating, price)

# Plot the linear regression line
plt.plot(review_score_rating, predicted_price, color='red')

# Set the title and axis labels
plt.title('Price vs. Review Score Rating')
plt.xlabel('Review Score Rating')
plt.ylabel('Price')

# Show the plot
plt.show()
```
- Here was the outcome from my data.

![priceVSscorerating](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week_2_Project/priceVSscorerating.png)
- From the  the image there is a strong positive correlation between review score rating and price in the Airbnb.data.csv 
file. This means that Airbnb listings with higher review scores ratings tend to be more expensive.
 However, the relationship is non-linear, meaning that the rate at which the price increases as the review score ratng increases is not constant.

![priceVsdistance](https://github.com/edinabwari/Data_Science_For_Everyone_Projects/blob/main/Week_2_Project/priceVSdistance.png)

- Here also  there is a positive correlation between distance and price in the Airbnb.data.csv file. This means that Airbnb listings 
that are further away from the city center tend to be more expensive this relationship  is non-linear, thus not constant.

**