from sales_forecast import forecast_model
from numpy import linspace
import seaborn as sns
import matplotlib.pyplot as plt

model = forecast_model()


def plot_time_serie(model, index):
    data = model.train_data[model.train_data.ts_id == index]
    sns.set_style("darkgrid")
    sns.lineplot(x="Date", y="Sales", data=data)
    plt.xticks(ticks=linspace(0, len(data.Date), 10), rotation=15)
    plt.title("Time-Serie of index {}".format(index))
    plt.show()

plot_time_serie(model, 10)