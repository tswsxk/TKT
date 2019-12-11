# coding: utf-8
# create by tongshiwei on 2019/6/25

from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-pep8>=1',
    'EduData>=0.0.4',
]

try:
    import torch
    torch_requires = []
except ModuleNotFoundError:
    torch_requires = ["torch"]

setup(
    name='TKT',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    long_description='Refer to full documentation https://github.com/bigdata-ustc/TKT/blob/master/README.md'
                     ' for detailed information.',
    description='This project aims to '
                'provide multiple knowledge tracing models.',
    extras_require={
        'test': test_deps,
    },
    install_requires=torch_requires + [
        'tqdm',
        'mxnet',
        'gluonnlp',
        'sklearn',
        'longling>=1.3.2',
    ],  # And any other dependencies foo needs
    entry_points={
    },
)
