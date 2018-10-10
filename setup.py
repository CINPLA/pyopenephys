from setuptools import setup, find_packages
# import pyopenephys

long_description = open("README.md").read()

entry_points = None

setup(
    name="pyopenephys",
    version='0.1.1',
    author="Alessio Buccino",
    author_email="alessiob@ifi.uio.no",
    description="Python package for parsing Open Ephys data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CINPLA/py-open-ephys",
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'quantities',
        'xmltodict',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ]
)
