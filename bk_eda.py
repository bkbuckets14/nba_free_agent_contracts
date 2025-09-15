import matplotlib.pyplot as plt
import pandas as pd

def descriptive_stats_single_numeric(data):
    describe = data.describe()
    
    rnge = describe[7] - describe[3]
    iqr = describe[6] - describe[4]
    cov = describe[2]/describe[1]
    qcv = iqr/describe[5]
    
    temp = pd.Series(data={'range': rnge, 'iqr': iqr, 'cov': cov, 'qcv': qcv})
    
    describe = describe.append(temp)
    
    return pd.DataFrame(describe, columns=['Stats'])

def describe_by_category(data, numeric, categorical):
    grouped = data.groupby(categorical)
    grouped_y = grouped[numeric].describe()

    return grouped_y

def multiboxplot(data, numeric, categorical, avg_dollars=False):
    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    grouped = data.groupby(categorical)
    labels = pd.unique(data[categorical].values)
    labels.sort()
    grouped_data = [grouped[numeric].get_group( k) for k in labels]
    patch = axes.boxplot( grouped_data, labels=labels, patch_artist=True, zorder=1,
                         boxprops=dict(facecolor='w', color='black'), medianprops=dict(color='black'))

    if avg_dollars==True:
        axes.set_yticks([0, 10000000, 20000000, 30000000, 40000000])
        axes.set_yticklabels(['$0', '$10,000,000', '$20,000,000', '$30,000,000', '$40,000,000'])

    axes.set_xlabel(categorical)
    axes.set_ylabel(numeric)
    axes.set_title("Distribution of {0} by {1}".format(numeric, categorical))

    plt.show()
    plt.close()


