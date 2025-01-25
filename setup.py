from setuptools import setup

setup(
    name="pyBlockGrid",
    version="0.1.0",
    # ... other configurations ...
    extras_require={
        'test': [
            'pytest>=6.0',
            'pytest-cov',
            'numpy',
            'matplotlib',
            'scipy'
        ],
    },
) 