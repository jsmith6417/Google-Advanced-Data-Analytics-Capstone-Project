# Import the packages we need for the analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset into dataframe titled df
df = pd.read_csv('waze_dataset.csv')

# Display the first 10 rows of the dataframe (for a quick inspection)
df.head(10)

# Display the number of rows (records) in the dataframe
df.size

# Generate summary stastics
df.describe()

# Display basic dataframe info (column names, data types, etc.)
df.info()

# Display a box plot of the number of users opening the app (sessions) during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['sessions'], fliersize=1)
plt.title('sessions box plot');

# Display a histogram of the number of users opening the app (sessions) during the month that also shows the median
plt.figure(figsize=(5,3))
sns.histplot(x=df['sessions'])
median = df['sessions'].median()
plt.axvline(median, color='red', linestyle='--')
plt.text(75,1200, 'median=56.0', color='red')
plt.title('sessions box plot');

# Function to define histogram plots based on the format of the `sessions` histogram
def histogrammer(column_str, median_text=True, **kwargs):    # **kwargs = any keyword arguments
                                                             # from the sns.histplot() function
    median=round(df[column_str].median(), 1)
    plt.figure(figsize=(5,3))
    ax = sns.histplot(x=df[column_str], **kwargs)            # Plot the histogram
    plt.axvline(median, color='red', linestyle='--')         # Plot the median line
    if median_text==True:                                    # Add median text unless set to False
        ax.text(0.25, 0.85, f'median={median}', color='red',
            ha='left', va='top', transform=ax.transAxes)
    else:
        print('Median:', median)
    plt.title(f'{column_str} histogram');

# Display a histogram for the number of drives
histogrammer('drives')

# Display a box plot showing the number of app sessions since user onboarded
plt.figure(figsize=(5,1))
sns.boxplot(x=df['total_sessions'], fliersize=1)
plt.title('total_sessions box plot');

# Display a histogram showing the number of app sessions since user onboarded
histogrammer('total_sessions')

# Display a box plot showing the number of days since a user signed up for the app
plt.figure(figsize=(5,1))
sns.boxplot(x=df['n_days_after_onboarding'], fliersize=1)
plt.title('n_days_after_onboarding box plot');

# Display a histogram showing the number of days since a user signed up for the app
histogrammer('n_days_after_onboarding', median_text=False)
# Median text off

# Display a box plot of km driven during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driven_km_drives'], fliersize=1)
plt.title('driven_km_drives box plot');

# Display a histogram of km driven during the month
histogrammer('driven_km_drives')

# Display a box plot showing number of times the app is opened during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['activity_days'], fliersize=1)
plt.title('activity_days box plot');

# Display a histogram showing number of times the app is opened during the month
histogrammer('activity_days', median_text=False, discrete=True)
# Median text off
# Discrete is on (use the unique values in the dataset)

# Display a box plot showing the number of users who drive at least 1 km during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driving_days'], fliersize=1)
plt.title('driving_days box plot');

# Display a histogram showing the number of users who drive at least 1 km during the month
histogrammer('driving_days', median_text=False, discrete=True)
# Median text off
# Discrete is on (use the unique values in the dataset)

# Display a pie chart showing the number and percentage of Android and iPhone users
fig = plt.figure(figsize=(3,3))
data=df['device'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Users by device');

# Display a pie chart showing the number and percentage of retained and churned users
fig = plt.figure(figsize=(3,3))
data=df['label'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Count of retained vs. churned');

# Display a histogram showing driving days and activity days
plt.figure(figsize=(12,4))
label=['driving days', 'activity days']
plt.hist([df['driving_days'], df['activity_days']],
         bins=range(0,33),
         label=label)
plt.xlabel('days')
plt.ylabel('count')
plt.legend()
plt.title('driving_days vs. activity_days');

# Confirm the max values for driving days and app activity days
print(df['driving_days'].max())
print(df['activity_days'].max())

# Display a scatter plot showing driving days and activity days
sns.scatterplot(data=df, x='driving_days', y='activity_days')
plt.title('driving_days vs. activity_days')
plt.plot([0,31], [0,31], color='red', linestyle='--');

# Display a histogram of retention and churn by device
plt.figure(figsize=(5,4))
sns.histplot(data=df,
             x='device',
             hue='label',
             multiple='dodge',
             shrink=0.9
             )
plt.title('Retention by device histogram');

# Create a new column for km driven per driving day
# 1. Create `km_per_driving_day` column
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Display descriptive statistics for the new column
df['km_per_driving_day'].describe()

# Convert infinite values to zero
# 1. Convert infinite values to zero
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0

# 2. Display descriptive statistics to make sure the change worked as expected
df['km_per_driving_day'].describe()

# Display a stacked histogram of churn rate by mean km driven
plt.figure(figsize=(12,5))
sns.histplot(data=df,
             x='km_per_driving_day',
             bins=range(0,1201,20),
             hue='label',
             multiple='fill')
plt.ylabel('%', rotation=0)
plt.title('Churn rate by mean km per driving day');

# Display a stacked histogram of churn rate by driving days
plt.figure(figsize=(12,5))
sns.histplot(data=df,
             x='driving_days',
             bins=range(1,32),
             hue='label',
             multiple='fill',
             discrete=True)
plt.ylabel('%', rotation=0)
plt.title('Churn rate per driving day');

# Create a new column showing the proportion of sessions that occured in the last month
df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# Display the median value of the new of the new column
df['percent_sessions_in_last_month'].median()

# Display a histogram of the vlues in the new column
histogrammer('percent_sessions_in_last_month',
             hue=df['label'],
             multiple='layer',
             median_text=False)

# Check the median value of n_days_after_onboarding
df['n_days_after_onboarding'].median()

# Display a histogram of n_days_after_onboarding values for users with at least 40% sessions in the last month
data = df.loc[df['percent_sessions_in_last_month']>=0.4]
plt.figure(figsize=(5,3))
sns.histplot(x=data['n_days_after_onboarding'])
plt.title('Num. days after onboarding for users with >=40% sessions in last month');

# Define the outlier threshold
def outlier_imputer(column_name, percentile):
    # Calculate threshold
    threshold = df[column_name].quantile(percentile)
    # Impute threshold for values > than threshold
    df.loc[df[column_name] > threshold, column_name] = threshold

    print('{:>25} | percentile: {} | threshold: {}'.format(column_name, percentile, threshold))

# Find the outliers
for column in ['sessions', 'drives', 'total_sessions',
               'driven_km_drives', 'duration_minutes_drives']:
               outlier_imputer(column, 0.95)

# Generate descriptive statistics for the dataframe to make sure the changes worked as expected
df.describe()

# Choose the plots that best make the point for an executive summary
# Prepare the exexcutive summary