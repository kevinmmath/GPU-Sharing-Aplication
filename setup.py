from setuptools import setup, find_packages

setup(
    name="fake_cuda",
    version="0.1.0",
    description="Phase 1 — GPU Sharing: intercept and log every PyTorch op",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
        "network": ["pyzmq>=25.0.0"],   # Phase 2
    },
)
