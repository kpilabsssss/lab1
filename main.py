import numpy as np
import matplotlib.pyplot as plt

class Lab1:
    def __init__(self, sample_size=117):
        """
        self.mu - среднее значение нормального распределения
        self.sigma - стандартое  отклонение нормального распределения
        self.sample - выборка
        """
        self.mu, self.sigma = np.random.normal(loc=0, scale=1, size=2)
        while self.sigma <= 0:
            self.mu, self.sigma = np.random.normal(loc=0, scale=1, size=2)
        self.sample = np.random.normal(loc=self.mu, scale=self.sigma, size=sample_size)

    def plot_histogram(self, bins=10, density=True, alpha=0.6, color='b', edgecolor='black'):
        """
        строит гистограму для выборки
        """
        plt.hist(self.sample, bins=bins, density=density, alpha=alpha, color=color, edgecolor=edgecolor)

    def plot_polygon(self):
        """
        строит полигон для выборки
        """
        plt.plot(sorted(self.sample), np.linspace(0, 1, len(self.sample), endpoint=False), 'r--')

    def set_labels(self, title='Нормальний розподіл', xlabel='Значення', ylabel='Ймовірність'):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def show(self):
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
    test.plot_histogram()
    test.plot_polygon()
    test.set_labels()
    test.show()
    print(f'выборочное среднее: {test.sample_mean()}')
    print(f'медиана: {test.sample_median()}')
    print(f'мода: {test.sample_mode()}')
    print(f'дисперсия: {test.sample_variance()}')
    print(f'среднеквадратичное отклонение: {test.sample_standard_deviation()}')