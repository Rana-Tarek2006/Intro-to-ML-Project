from setuptools import setup, find_packages

setup(
    name='market-trend-identification',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'yfinance>=0.2.0',
    ],
)