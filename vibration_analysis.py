import numpy as np
from scipy.fft import fft
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# 振動數據分析類別


class VibrationAnalysis:
    def __init__(self, input_size, hidden_size, num_layers, output_size, time_steps=1000, base_frequency=1.0, sequence_length=10):
        """
        初始化振動分析類別，定義網路結構和參數。
        參數:
        - input_size: 輸入資料的大小
        - hidden_size: 隱藏層大小
        - num_layers: 神經網路的層數
        - output_size: 輸出資料大小
        - time_steps: 時間步長
        - base_frequency: 基頻
        - sequence_length: 序列長度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.time_steps = time_steps
        self.base_frequency = base_frequency
        self.sequence_length = sequence_length

    def generate_complex_vibration_data(self, harmonics=3, noise_level=0.1, wear_factor=0.05):
        """
        生成模擬的複雜振動數據，包含基礎頻率和諧波。
        參數:
        - harmonics: 諧波的數量
        - noise_level: 隨機噪聲的強度
        - wear_factor: 設備磨損導致的振幅增加
        返回:
        - 振動數據 (numpy array)
        """
        t = np.linspace(0, 1, self.time_steps)
        vibration_data = np.sin(2 * np.pi * self.base_frequency * t)  # 基礎頻率

        # 添加諧波
        for i in range(2, harmonics + 1):
            vibration_data += (1 / i) * np.sin(2 * np.pi *
                                               i * self.base_frequency * t)

        # 添加隨機噪聲
        noise = noise_level * np.random.normal(size=t.shape)
        vibration_data += noise

        # 模擬設備磨損
        wear_effect = wear_factor * np.linspace(1, 1.5, self.time_steps)
        vibration_data *= wear_effect

        return vibration_data

    def calculate_time_domain_features(self, vibration_data):
        """
        計算時間域特徵。
        參數:
        - vibration_data: 振動數據 (numpy array)
        返回:
        - 特徵向量 (字典)
        """
        rms_value = np.sqrt(np.mean(vibration_data**2))
        peak_value = np.max(vibration_data)
        kurtosis_value = stats.kurtosis(vibration_data)
        skewness_value = stats.skew(vibration_data)
        crest_factor = peak_value / rms_value if rms_value != 0 else 0

        return {
            'RMS': rms_value,
            'Peak': peak_value,
            'Kurtosis': kurtosis_value,
            'Skewness': skewness_value,
            'Crest Factor': crest_factor
        }

    def calculate_frequency_domain_features(self, vibration_data):
        """
        計算頻率域特徵。
        參數:
        - vibration_data: 振動數據 (numpy array)
        返回:
        - 特徵向量 (字典)
        """
        fft_result = fft(vibration_data)
        magnitude_spectrum = np.abs(fft_result)[:len(fft_result)//2]  # 取前半部

        # 計算主導頻率
        dominant_freq = np.argmax(magnitude_spectrum)

        # 計算頻譜質心
        freq_bins = np.fft.fftfreq(len(vibration_data), d=1/self.time_steps)
        centroid = np.sum(freq_bins[:len(fft_result)//2]
                          * magnitude_spectrum) / np.sum(magnitude_spectrum)

        # 計算頻譜熵
        prob_density = magnitude_spectrum / np.sum(magnitude_spectrum)  # 概率密度
        # 加上小常數避免對數計算出現問題
        entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))

        return {
            'Dominant Frequency': dominant_freq,
            'Spectral Centroid': centroid,
            'Spectral Entropy': entropy
        }

    def simulate_health_index_degradation(self, initial_health=1.0, degradation_rate=0.1, time_steps=100):
        """
        模擬設備健康指數的漸進式退化。
        參數:
        - initial_health: 初始健康指數
        - degradation_rate: 健康指數退化速率
        - time_steps: 時間步數
        返回:
        - 健康指數 (numpy array)
        """
        health_index = np.zeros(time_steps)
        for t in range(time_steps):
            health_index[t] = initial_health * \
                np.exp(-degradation_rate * t) + np.random.normal(0, 0.01)

        return health_index

    def normalize_data(self, data):
        """
        將數據進行歸一化。
        參數:
        - data: 原始數據 (numpy array)
        返回:
        - 歸一化後的數據
        """
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return normalized_data

    def generate_complex_vibration_data(self, harmonics=3, noise_level=0.1, wear_factor=0.05):
        """
        生成模擬的複雜振動數據，包含基礎頻率和諧波。
        參數:
        - harmonics: 諧波的數量
        - noise_level: 隨機噪聲的強度
        - wear_factor: 設備磨損導致的振幅增加
        返回:
        - 振動數據 (numpy array)
        """
        t = np.linspace(0, 1, self.time_steps)
        vibration_data = np.sin(2 * np.pi * self.base_frequency * t)  # 基礎頻率

        # 添加諧波
        for i in range(2, harmonics + 1):
            vibration_data += (1 / i) * np.sin(2 * np.pi *
                                               i * self.base_frequency * t)

        # 添加隨機噪聲
        noise = noise_level * np.random.normal(size=t.shape)
        vibration_data += noise

        # 模擬設備磨損
        wear_effect = wear_factor * np.linspace(1, 1.5, self.time_steps)
        vibration_data *= wear_effect

        return vibration_data
