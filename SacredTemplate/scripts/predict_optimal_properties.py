import pandas as pd


def main():
    df= pd.read_csv('database.csv')
    X = df.iloc[:, :8]
    Y = df.iloc[:, 25:28]



if __name__ == "__main__":
    main()