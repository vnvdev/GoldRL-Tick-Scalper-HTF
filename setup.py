from setuptools import setup, find_packages

setup(
    name="xau_rl_scalp",
    version="1.0.0",
    description="99.99% Real Market Standard Tick Scalping RL Environment for Exness XAUUSD Zero Account",
    author="Kudzo Vu (Vietnam Quant & Algorithmic Trading Developer - fb.com/kudzovu - t.me/kudzovu)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "polars>=0.20.0",
        "numba>=0.58.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.3.0",
        "gymnasium>=0.29.0",
    ],
    python_requires=">=3.10",
)
