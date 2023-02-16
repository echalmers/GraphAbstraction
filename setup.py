from setuptools import setup

setup(
    name="GraphAbstraction",
    version="0.0.1",
    packages=['GraphAbstraction'],
    install_requires=[
        "hippocluster @ git+https://github.com/echalmers/hippocluster.git",
        "matplotlib",
        "numpy",
        "networkx",
    ]
)
