from common import read_input_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = read_input_data()
    print(df.size)
    plt(df)
    print('Hello World')