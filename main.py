import numpy as np
import matplotlib.pyplot as plt

class Lab1:
    def __init__(self, sample_size=117):
        """
        self.mu - среднее значение нормального распределения
        self.sigma - стандартое  отклонение нормального распределения
        self.sample - выборка
        """
        self.mu, self.sigma = np.random.normal(loc=0, scale=1, size=2) * 1.2
        while self.sigma <= 0:
            self.mu, self.sigma = np.random.normal(loc=0, scale=1, size=2) * 1.2
        self.sample = np.random.normal(loc=self.mu, scale=self.sigma, size=sample_size)

    def plot(self, bins=10, density=True, alpha=0.6, color='b', edgecolor='black',
             title='нормальний розподіл', xlabel='значення', ylabel='ймовірність'):
        """
        строит гистограмму и полигон для выборки
        """
        plt.hist(self.sample, bins=bins, density=density, alpha=alpha, color=color, edgecolor=edgecolor)
        plt.plot(sorted(self.sample), np.linspace(0, 1, len(self.sample), endpoint=False), 'r--')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_box(self, color='b', title='діаграма розмаху', ylabel='значення'):
        """
        строит диаграмму розмаху для выборки
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
        строит диаграмму Парето для выборки
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
        строит круговую диаграмму для выборки
        """
        counts, bins = np.histogram(self.sample, bins=bins)
        labels = [f'{bins[i]:.2f} - {bins[i+1]:.2f}' for i in range(len(bins)-1)]
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title('кругова діаграма')
        plt.show()

    def sample_mean(self):
        """
        выборочное среднее
        """
        return np.sum(self.sample) / len(self.sample)

    def sample_median(self):
        """
        медиана
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
        дисперсия
        """
        sample_mean_value = self.sample_mean()
        return np.sum((self.sample - sample_mean_value) ** 2) / (len(self.sample) - 1)

    def sample_standard_deviation(self):
        """
        стандартное отклонение
        """
        return np.sqrt(self.sample_variance())

if __name__ == '__main__':
    test = Lab1()
    test.plot()
    test.plot_box()
    test.plot_pareto()
    test.plot_pie()
    print(f'выборочное среднее: {test.sample_mean()}')
    print(f'медиана: {test.sample_median()}')
    print(f'мода: {test.sample_mode()}')
    print(f'дисперсия: {test.sample_variance()}')
    print(f'среднеквадратичное отклонение: {test.sample_standard_deviation()}')
