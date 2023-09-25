from setuptools import setup, find_packages

setup(
    name="portfolio_construction_toolkit",
    version="1.0.0",
    packages=find_packages(),
    description = 'Portfolio construction and analysis. Find the optimal weights and backtest different kinds of portfolios.',
    author =  'Titiksh Prakash',
    keywords=['Portfolio Optimization', 'Portfolio Analysis', 'Portfolio Construction', 'Portfolio Backtesting', 'Mordern Portfolio Theory'],
    python_requires='>=3',
    install_requires=[
        "pandas>=2.0.2",
        "numpy>=1.25.0",
        "matplotlib>=3.7.1",
        "yfinance>=0.2.22",
        "scipy>=1.11.0"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'
    ],
)