from sales_forecast import forecast_model
from numpy import linspace
import seaborn as sns
import matplotlib.pyplot as plt

model = forecast_model()

sns.set_style("darkgrid")


def plot_time_serie(model, index):
    data = model.train_data[model.train_data.ts_id == index]
    sns.lineplot(x="Date", y="Sales", data=data)
    plt.xticks(ticks=linspace(0, len(data.Date), 10), rotation=15)
    plt.title("Time-Serie of index {}".format(index))
    plt.show()


def plot_hist_sku(model, feature, hue=None):
    """Plot the histogram representing the distribution of sku across the features [Segment, Pack, Product], and cross it with another feature as color"""
    if feature not in ["Segment", "Pack", "Product"]:
        raise ValueError("""Incorrect feature name. Should be one of these : "Segment", "Pack", "Product".""")
    if hue not in ["Segment", "Pack", "Product"]:
        raise ValueError("""Incorrect hue name. Should be one of these : "Segment", "Pack", "Product".""")
    sns.histplot(data=model.features, y=feature, hue=hue)
    plt.title("Types of " + feature)
    plt.show()

#plot_time_serie(model, 10)

plot_hist_sku(model, "Segment", hue="Pack")
plot_hist_sku(model, "Pack", hue="Product")
plot_hist_sku(model, "Product", hue="Segment")
