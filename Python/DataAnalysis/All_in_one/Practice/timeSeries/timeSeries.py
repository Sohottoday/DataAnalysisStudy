# 시계열 데이터
"""
시계열 데이터의 특징

시계열에서 반드시 고려해야 할 사항
원계열 = Trend + Cycle(계절요인 등) + 불규칙 term
"""

# Monthly Data
import numpy as np
import pandas as pd
import matplotlib.pyplt as plt
from statsmodels.tsa.seasonal import seasonal_decompose