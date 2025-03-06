import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


def plot_relational_plot(df):
    """ Create a barplot """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 6))
        sns.barplot(df[numeric_cols])
        plt.suptitle('Relational Plot for Numeric Columns', y=1.02)
        plt.savefig('relational_plot.png')
        plt.show()
    else:
        print("Not enough numeric columns to plot relational graph.")


def plot_categorical_plot(df):
    """ create a categirical plot"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df, x=categorical_cols[0], y=numeric_cols[0]
        )
        plt.xticks(rotation=90)
        plt.title(f'Categorical Plot: {categorical_cols[0]} vs {numeric_cols[0]}')
        plt.savefig('categorical_plot.png')
        plt.show()


def plot_statistical_plot(df):
    """ Create a statisical plot """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[numeric_cols])
        plt.xticks(rotation=90)
        plt.title('Statistical Plot for Numeric Columns')
        plt.savefig('statistical_plot.png')
        plt.show()


def statistical_analysis(df, col):
    """perform statistical analysis on a column """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col].dropna())
    excess_kurtosis = ss.kurtosis(df[col].dropna())
    
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """ preprocess and analyze dataset """
    print("Dataset Overview:")
    print(df.info())
    print(df.describe(include='all'))
    print(df.head())
    
    print("Correlation Matrix:")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print(numeric_df.corr())
        
    return df


def writing(moments, col):
    """print stastical details in a formatted manner"""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew_desc = "not skewed" if abs(moments[2]) < 2 else ("right skewed" if moments[2] > 0 else "left skewed")
    kurt_desc = "mesokurtic" if abs(moments[3]) < 2 else ("leptokurtic" if moments[3] > 0 else "platykurtic")
    print(f'The data was {skew_desc} and {kurt_desc}.')

def main():
    df = pd.read_csv("data.csv")
    
    # Convert any numeric columns stored as strings to numeric values
    df = df.apply(pd.to_numeric, errors='ignore')
    df = preprocessing(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        print("No numerical columns found in the dataset.")
        return
    
    plot_relational_plot(df)
    
    plot_statistical_plot(df)
    
    plot_categorical_plot(df)
    
    for col in numeric_cols:
        moments = statistical_analysis(df, col)
        writing(moments, col)

if __name__ == '__main__':
    main()
