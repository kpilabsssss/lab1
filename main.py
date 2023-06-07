import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2

class Lab1:
    def __init__(self, sample_size=117):
        """
        self.mu - середнє значення нормального розподілу
        self.sigma - стандартне відхилення нормального розподілу
        self.sample - вибірка
        """
        self.mu, self.sigma = np.random.normal(loc=0, scale=1, size=2) * 1.2
        while self.sigma <= 0:
            self.mu, self.sigma = np.random.normal(loc=0, scale=1, size=2) * 1.2
        self.sample = np.random.normal(loc=self.mu, scale=self.sigma, size=sample_size)

    def plot(self, bins=10, density=True, alpha=0.6, color='b', edgecolor='black',
             title='нормальний розподіл', xlabel='значення', ylabel='ймовірність'):
        """
        будує гістограму та полігон для вибірки
        """
        plt.hist(self.sample, bins=bins, density=density, alpha=alpha, color=color, edgecolor=edgecolor)
        plt.plot(sorted(self.sample), np.linspace(0, 1, len(self.sample), endpoint=False), 'r--')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_box(self, color='b', title='діаграма розмаху', ylabel='значення'):
        """
        будує діграму розмаху для вибірки
        """
        plt.boxplot(self.sample, vert=False, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=color, color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    medianprops=dict(color='red'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='black'))
        plt.title(title)
        plt.ylabel(ylabel)
        plt.show()

    def plot_pareto(self, xlabel='Значення', ylabel='Кумулятивна частота'):
        """
        будує діаграму Парето для вибірки
        """
        sorted_sample = np.sort(self.sample)[::-1]
        cum_freq = np.cumsum(sorted_sample) / np.sum(sorted_sample)
        fig, ax1 = plt.subplots()
        ax1.bar(np.arange(len(sorted_sample)), sorted_sample, color='b')
        ax1.set_xlabel('значення')
        ax1.set_ylabel('частота', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(cum_freq, 'r--')
        ax2.set_ylabel('кумулятивна частота', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title('діаграма парето')
        plt.show()

    def plot_pie(self, bins=10):
        """
        будує кругову діаграму для вибірки
        """
        counts, bins = np.histogram(self.sample, bins=bins)
        labels = [f'{bins[i]:.2f} - {bins[i+1]:.2f}' for i in range(len(bins)-1)]
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title('кругова діаграма')
        plt.show()

    def sample_mean(self):
        """
        вибіркове середнє
        """
        return np.sum(self.sample) / len(self.sample)

    def sample_median(self):
        """
        медіана
        """
        sorted_sample = sorted(self.sample)
        n = len(self.sample)
        if n % 2 == 0: return (sorted_sample[n // 2 - 1] + sorted_sample[n // 2]) / 2
        else: return sorted_sample[n // 2]

    def sample_mode(self):
        """
        мода
        """
        min_val = np.min(self.sample)
        counts = np.bincount((self.sample - min_val + 1).astype(int))
        return np.argmax(counts) + min_val - 1

    def sample_variance(self):
        """
        дисперсія
        """
        sample_mean_value = self.sample_mean()
        return np.sum((self.sample - sample_mean_value) ** 2) / (len(self.sample) - 1)

    def sample_standard_deviation(self):
        """
        стандартне відхилення
        """
        return np.sqrt(self.sample_variance())

    def confidence_interval_mean(self, confidence_level=0.95):
        n = len(self.sample)
        dof = n - 1
        t_value = t.ppf((1 + confidence_level) / 2, dof)
        std_error = self.sample_standard_deviation() / np.sqrt(n)
        lower_bound = self.sample_mean() - t_value * std_error
        upper_bound = self.sample_mean() + t_value * std_error
        return (lower_bound, upper_bound)

    def confidence_interval_std(self, confidence_level=0.95):
        n = len(self.sample)
        chi2_value_1 = chi2.ppf((1 + confidence_level) / 2, n - 1)
        chi2_value_2 = chi2.ppf((1 - confidence_level) / 2, n - 1)
        lower_bound = np.sqrt((n - 1) * self.sample_variance() / chi2_value_1)
        upper_bound = np.sqrt((n - 1) * self.sample_variance() / chi2_value_2)
        return (lower_bound, upper_bound)

if __name__ == '__main__':
    test = Lab1()
    test.plot()
    test.plot_box()
    test.plot_pareto()
    test.plot_pie()
    print(f'вибіркове середнє: {test.sample_mean()}')
    print(f'медіана: {test.sample_median()}')
    print(f'мода: {test.sample_mode()}')
    print(f'дисперсія: {test.sample_variance()}')
    print(f'середньоквадратичне відхилення: {test.sample_standard_deviation()}')
    print(f'довірчий інтервал для середнього (95%): {test.confidence_interval_mean()}')
    print(f'довірчий інтервал для стандартного відхилення (95%): {test.confidence_interval_std()}')

    sample_sizes = [50, 100, 150, 200]
    confidence_levels = [0.90, 0.95, 0.99]
    means = []
    variances = []
    mean_intervals = []
    variance_intervals = []

    for sample_size in sample_sizes:
        lab = Lab1(sample_size=sample_size)
        means.append(lab.sample_mean())
        variances.append(lab.sample_variance())
        mean_intervals_row = []
        variance_intervals_row = []
        for confidence_level in confidence_levels:
            mean_interval = lab.confidence_interval_mean(confidence_level=confidence_level)
            variance_interval = lab.confidence_interval_std(confidence_level=confidence_level)
            mean_intervals_row.append(mean_interval)
            variance_intervals_row.append(variance_interval)
        mean_intervals.append(mean_intervals_row)
        variance_intervals.append(variance_intervals_row)

    plt.plot(sample_sizes, means, marker='o')
    plt.xlabel('обсяг вибірки')
    plt.ylabel('вибіркове середнє')
    plt.title('залежність вибіркового середнього від обсягу вибірки')
    plt.show()

    plt.plot(sample_sizes, variances, marker='o')
    plt.xlabel('обсяг вибірки')
    plt.ylabel('дисперсія')
    plt.title('залежність дисперсії від обсягу вибірки')
    plt.show()

    table_mean_intervals = np.array(mean_intervals)
    plt.table(cellText=table_mean_intervals, rowLabels=sample_sizes, colLabels=confidence_levels, loc='center')
    plt.title('довірчі інтервали для середнього')
    plt.axis('off')
    plt.show()

    table_variance_intervals = np.array(variance_intervals)
    plt.table(cellText=table_variance_intervals, rowLabels=sample_sizes, colLabels=confidence_levels, loc='center')
    plt.title('довірчі інтервали для дисперсії')
    plt.axis('off')
    plt.show()