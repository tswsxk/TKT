# coding: utf-8
# create by tongshiwei on 2019/6/25

from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
    'EduData>=0.0.17',
]
bench_deps = [
    'EduData>=0.0.17',
    'fire'
]

try:
    import torch
    torch_requires = []
except ModuleNotFoundError:
    torch_requires = ["torch"]

setup(
    name='TKT',
    version='0.0.2',
    packages=find_packages(),
    python_requires='>=3.6',
    long_description='Refer to full documentation https://github.com/bigdata-ustc/TKT/blob/master/README.md'
                     ' for detailed information.',
    description='This project aims to '
                'provide multiple knowledge tracing models.',
    extras_require={
        'test': test_deps,
        'benchmark': bench_deps,
    },
    install_requires=torch_requires + [
        'EduKTM>=0.0.6',
        'PyBaize>=0.0.4',
    ],  # And any other dependencies foo needs
    entry_points={
    },
)
