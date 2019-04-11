# Description
Tealeaves is a hands-off plotting tool that lets you leverage some useful visualization types with a minimal amount of code. The idea is that you simply provide a data frame and tealeaves does the rest for you.

# Examples
## Distribution comparison plot
Compare the distributions of columns of two dataframes in ways that make sense for the datatype and the properties of the data. For example a categorical variable would be plotted as a bar chart, unless there is a very high number of categories in which case the number of categories present in dataset A, B or BOTH is shown.
```
titanic, train, test = load_demo_data()
dist_compare_grid(train, test, grid_size=(950, 712.5))
```
![Image failed to load](https://raw.githubusercontent.com/atomichighfive/tealeaves/master/tealeaves/demos/sample_images/demo_dist_compare_plot.png)

## Dataframe plot
Plot a dataframe highlighting unusual, missing or extreme values. This is a good way of figuring out the quality of data you just got your hands on.
```
titanic, train, test = load_demo_data()
dataframe_plot(
    titanic, rows_per_pixel=1, extra_test_dict={13: np.array([0, 0, 255]) / 255}
)
```
![Image failed to load](https://raw.githubusercontent.com/atomichighfive/tealeaves/master/tealeaves/demos/sample_images/demo_dataframe_plot.png)

## Index-value plot
Plot values agains indices. This gives you an idea of how the data is sorted and how the order of one variable translates to the order of another variable. By providing test data you can also tell if the test data is drawn randomly or as a block from an ordered data set.
```
titanic, train, test = load_demo_data()
    index_value_plot(
        train,
        test,
        target='Survived'
    )
```
![Image failed to load](https://raw.githubusercontent.com/atomichighfive/tealeaves/master/tealeaves/demos/sample_images/demo_index_value_plot.png)

# Installation
`pip install git+git://github.com/atomichighfive/tealeaves`
