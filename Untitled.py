#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


data = pd.read_csv('C:/Users/abrah/Downloads/COP4612_HW1.csv')


# In[4]:


#Question 1
data[0:10]


# In[5]:


#Question 2
print(data.columns)


# In[6]:


#Question 3
print(data.dtypes)


# In[7]:


#Question 4
correlation_matrix = data.corr()
print(correlation_matrix)


# In[8]:


#Question 5
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.pairplot(data)
plt.show()


# In[10]:


#Question 6
import pandas as pd

data = pd.get_dummies(data, columns=['sex', 'position', 'pay_grade'])

print(data.columns)

print(data.head())


# In[11]:


#Question 7
import matplotlib.pyplot as plt

women_data = data[data['sex_F'] == 1]

plt.figure(figsize=(10, 6))
plt.scatter(women_data['age'], women_data['salary'], color='blue', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Salary vs. Age for Women')
plt.grid(True)
plt.show()


# In[14]:


#Question 8
import matplotlib.pyplot as plt

positions = ['Developer', 'Engineer', 'FinancialAnalyst', 'Secretary', 'Accountant']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

axes = axes.flatten()

for i, position in enumerate(positions):

    women_data_position = data[(data['sex_F'] == 1) & (data[f'position_{position}'] == 1)]

    axes[i].scatter(women_data_position['age'], women_data_position['salary'], color='blue', alpha=0.5)
    axes[i].set_xlabel('Age')
    axes[i].set_ylabel('Salary')
    axes[i].set_title(f'Salary vs. Age for Women in {position}')
    axes[i].grid(True)

plt.tight_layout()

plt.show()

# As observed in these graphs, older women earn the biggest amount of money in every career field. In other words, older women earn more than younger women.


# In[18]:


#Question 9
import matplotlib.pyplot as plt

women_data = data[data['sex_F'] == 1]

positions = ['Accountant', 'Developer', 'Engineer', 'FinancialAnalyst', 'Secretary']

colors = ['blue', 'green', 'red', 'purple', 'orange']

plt.figure(figsize=(10, 6))
for i, position in enumerate(positions):
    position_data = women_data[women_data[f'position_{position}'] == 1]
    plt.scatter(position_data['age'], position_data['salary'], color=colors[i], label=position, alpha=0.5)

plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Salary vs. Age for Women by Position')

plt.legend()

plt.grid(True)
plt.show()

# The women who earn the most money are 60 year-old accountants, developers and engineers.


# In[20]:


#Question 10
import pandas as pd

women_stat_df = pd.DataFrame(columns=['Position', 'Count', 'Min Salary', 'Max Salary', 'Mean Salary', 'Std Dev Salary', 'Avg Age'])

position = 'All'
count = women_data.count()[0]
min_salary = round(women_data['salary'].min(), 2)
max_salary = round(women_data['salary'].max(), 2)
mean_salary = round(women_data['salary'].mean(), 2)
std_dev_salary = round(women_data['salary'].std(), 2)
avg_age = round(women_data['age'].mean(), 2)

women_stat_df.loc[len(women_stat_df)] = [position, count, min_salary, max_salary, mean_salary, std_dev_salary, avg_age]

positions = ['Accountant', 'Developer', 'Engineer', 'FinancialAnalyst', 'Secretary']

for position in positions:
    count = women_data[women_data[f'position_{position}'] == 1].count()[0]
    min_salary = round(women_data[women_data[f'position_{position}'] == 1]['salary'].min(), 2)
    max_salary = round(women_data[women_data[f'position_{position}'] == 1]['salary'].max(), 2)
    mean_salary = round(women_data[women_data[f'position_{position}'] == 1]['salary'].mean(), 2)
    std_dev_salary = round(women_data[women_data[f'position_{position}'] == 1]['salary'].std(), 2)
    avg_age = round(women_data[women_data[f'position_{position}'] == 1]['age'].mean(), 2)

    women_stat_df.loc[len(women_stat_df)] = [position, count, min_salary, max_salary, mean_salary, std_dev_salary, avg_age]

print(women_stat_df)


# In[22]:


#Question 11
import matplotlib.pyplot as plt

def remove_outliers(data, columns, n_std):
    for col in columns:
        mean = data[col].mean()
        std_dev = data[col].std()
        threshold = n_std * std_dev
        data = data[(data[col] >= mean - threshold) & (data[col] <= mean + threshold)]
    return data

columns_to_check = ['salary', 'age']

n_std = 2.2

positions = ['All', 'Accountant', 'Developer', 'Engineer', 'FinancialAnalyst', 'Secretary']
for position in positions:
    if position == 'All':
        data_to_process = women_data
    else:
        data_to_process = women_data[women_data[f'position_{position}'] == 1]
    
    data_to_process = remove_outliers(data_to_process, columns_to_check, n_std)

    plt.figure(figsize=(10, 6))
    plt.scatter(data_to_process['age'], data_to_process['salary'], color='blue', alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.title(f'Salary vs. Age for Women in {position} (With Outliers)')
    plt.grid(True)
    plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data_to_process['age'], data_to_process['salary'], color='blue', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title(f'Salary vs. Age for Women in All Positions (Without Outliers)')
plt.grid(True)
plt.show()


# In[24]:


#Question 12
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

train_test_sizes = np.arange(0.2, 1.0, 0.2)

positions = ['All', 'Accountant', 'Developer', 'Engineer', 'FinancialAnalyst', 'Secretary']

feature_columns = ['age']

results = {}

for position in positions:
    if position == 'All':
        data_to_process = women_data
    else:
        data_to_process = women_data[women_data[f'position_{position}'] == 1]

    mse_list = []
    train_size_list = []

    for train_size in train_test_sizes:

        train_size_int = int(len(data_to_process) * train_size)
        train_data = data_to_process.head(train_size_int)
        test_data = data_to_process.tail(len(data_to_process) - train_size_int)

        model = LinearRegression()
        model.fit(train_data[feature_columns], train_data['salary'])

        predictions = model.predict(test_data[feature_columns])

        mse = mean_squared_error(test_data['salary'], predictions)

        mse_list.append(mse)
        train_size_list.append(train_size)

    position_results_data = pd.DataFrame({'Train Size': train_size_list, 'MSE': mse_list})
    position_results_data['Position'] = position
 
    results[position] = position_results_data

for position, data in results.items():
    print(f'Results for {position}:')
    print(data)
    print()


# In[25]:


#Question 13
import matplotlib.pyplot as plt

filtered_data = women_data[(women_data['sex_F'] == 1) & (women_data['age'] < 40)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

axes[0].scatter(women_data['age'], women_data['salary'], color='blue', alpha=0.5)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Salary')
axes[0].set_title('Original Data')
axes[0].grid(True)

axes[1].scatter(filtered_data['age'], filtered_data['salary'], color='red', alpha=0.5)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Salary')
axes[1].set_title('Filtered Data (Age < 40)')
axes[1].grid(True)

plt.tight_layout()

plt.show()

# The model may not be able to capture relationships that are specific to older age groups. If there are significant age-related trends that are not captured in the younger subset, the model's predictive accuracy may be affected.


# In[28]:


#Question 14
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

filtered_data = women_data[(women_data['sex_F'] == 1) & (women_data['age'] < 40)]

train_test_sizes = np.arange(0.2, 1.0, 0.2)

feature_columns = ['age']

results = []

for train_size in train_test_sizes:

    result_dict = {}
 
    train_size_int = int(len(filtered_data) * train_size)
    train_data = filtered_data.head(train_size_int)
    test_data = filtered_data.tail(len(filtered_data) - train_size_int)
 
    model = LinearRegression()
    model.fit(train_data[feature_columns], train_data['salary'])

    train_predictions = model.predict(train_data[feature_columns])
    test_predictions = model.predict(test_data[feature_columns])
    
    train_mse = mean_squared_error(train_data['salary'], train_predictions)
    test_mse = mean_squared_error(test_data['salary'], test_predictions)

    train_r_squared = model.score(train_data[feature_columns], train_data['salary'])
    test_r_squared = model.score(test_data[feature_columns], test_data['salary'])

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    model_variance = np.var(train_predictions)
    model_error = np.mean((train_data['salary'] - train_predictions) ** 2)

    avg_salary = filtered_data['salary'].mean()

    result_dict['Train'] = train_size
    result_dict['Test'] = 1 - train_size
    result_dict['Test_R_Score'] = test_r_squared
    result_dict['Test_RMSE'] = test_rmse
    result_dict['Train_R_Score'] = train_r_squared
    result_dict['Model_var'] = model_variance
    result_dict['Model_error'] = model_error
    result_dict['Avg_salary'] = avg_salary

    results.append(result_dict)

results_data = pd.DataFrame(results)

print("Results for Filtered Data (Age < 40):")
print(results_data)


# In[31]:


#Question 15
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize and train the model
model = LinearRegression()
model.fit(train_data[feature_columns], train_data['salary'])

# Predict on training and test sets
train_predictions = model.predict(train_data[feature_columns])
test_predictions = model.predict(test_data[feature_columns])

# Calculate R-squared scores
train_r_squared = model.score(train_data[feature_columns], train_data['salary'])
test_r_squared = model.score(test_data[feature_columns], test_data['salary'])

# Calculate Root Mean Squared Errors
train_rmse = np.sqrt(mean_squared_error(train_data['salary'], train_predictions))
test_rmse = np.sqrt(mean_squared_error(test_data['salary'], test_predictions))

# Get model variance
model_variance = np.var(train_predictions)

# Calculate mean of test error
test_error = test_data['salary'] - test_predictions
mean_test_error = np.mean(test_error)

# Calculate mean of train error
train_error = train_data['salary'] - train_predictions
mean_train_error = np.mean(train_error)

# Calculate average salary
avg_salary = filtered_data['salary'].mean()

# Print the specified metrics
print(f'Salary: {test_data["salary"].values}')
print(f'Test_R_Score: {test_r_squared}')
print(f'RMSE: {test_rmse}')
print(f'Training-Test Split: {best_train_size}')
print(f'Training_R_Score: {train_r_squared}')
print(f'Training RMSE: {train_rmse}')
print(f'Model Variance: {model_variance}')
print(f'Mean of Test Error: {mean_test_error}')
print(f'Mean of Train Error: {mean_train_error}')
print(f'Average Salary: {avg_salary}')

# Generate density vs salary graph (test error density)
plt.figure(figsize=(10, 6))
sns.histplot(test_error, kde=True, color='blue')
plt.xlabel('Test Error')
plt.ylabel('Density')
plt.title('Test Error Density vs Salary')
plt.show()

# Generate plot graph (test error)
plt.figure(figsize=(10, 6))
plt.scatter(test_data['salary'], test_error, color='red', alpha=0.5)
plt.xlabel('Salary')
plt.ylabel('Test Error')
plt.title('Test Error vs Salary')
plt.grid(True)
plt.show()


# In[ ]:




