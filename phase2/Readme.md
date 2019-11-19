Readme
======

Phase 2 of MWDB - CSE515 project. The dataset used for this project can be found [here][dataset_link].
This phase deals with dimension reductionality of vectors using techniques like SVD, PCA, and LDA 
and techniques such as CP-decomposition for tensors.

Installation
------------
* Python 3.6+
* Numpy 1.15.1
* Scipy 1.1.0
* [Pandas][pandas] 0.23.4
* [Scikit-learn][scikit] 0.20.0
* [tensorLy][tensorly] 0.4.2
* [seaborn][seaborn] 0.9.2
* [sqlite3][sqlite3] 3.23.1

conda python package manager is optional but highly recommended.

Usage
-----
Application is menu driven. To run the application from terminal run the
following :

	python main.py [Filepath to dataset]

This will create a sqlite file if the file is not already present there.

[dataset_link]: http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/
[pandas]: https://pandas.pydata.org/getpandas.html
[scikit]: http://scikit-learn.org/stable/
[tensorly]: http://tensorly.org/stable/index.html
[seaborn]: https://seaborn.pydata.org/
