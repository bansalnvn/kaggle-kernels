from common import read_input_data
import matplotlib.pyplot as plt


def filter_crop(crop_name, dataframe):
    filtered = dataframe[dataframe['Item Name'] == crop_name].filter(items=['Date', 'price'])
    return filtered


if __name__ == "__main__":
    plot_index = 14
    df, all_crops = read_input_data()
    df.set_index('Date')
    for idx, crop in enumerate(all_crops):
        filtered_dataframe = filter_crop(crop, df)
        print(filtered_dataframe.shape)
        if filtered_dataframe.shape[0] > 2047:
            print(idx, crop, filtered_dataframe)
            plt.plot(filtered_dataframe['Date'], filtered_dataframe['price'], label=crop)
            # filtered_dataframe.plot(x='Date', y='price', label=crop, marker='o')
    plt.show(block=True)
    plt.interactive(False)
