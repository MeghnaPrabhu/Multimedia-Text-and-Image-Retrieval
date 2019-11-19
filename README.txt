----------------------
SYSTEM REQUIREMENTS
----------------------
Python 3.6 with installed Pandas package
SQLite3
----------------------
Command Line Arguments
----------------------
The program works on dataset found at http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5
It takes a command line argument that specifies the location of the dataset in the sytem.
The program would fail if the dataset is not present at the specified location

----------------------
Functionality
----------------------
--The program has a Command Line interface and is designed to execute the below mentioned tasks
--It provides an interactive interface that prompts the user for inputs
--Press 1 to search text data and 2 for visual data
Tasks related to text data
1. Given a user ID, a model (TF, DF, TF-IDF), and value “k”, find the k most similar users  based on the text descriptors along with score. Also, find terms having highest similarity contribution. 
2. Given a image ID, a model (TF, DF, TF-IDF), and value “k”, find the k most similar images based on the text descriptors along with score. Also, find terms having highest similarity contribution. 
3. Given a location ID, a model (TF, DF, TF-IDF), and value “k”, find the k most similar locations based on the text descriptors along with score. Also, find terms having highest similarity contribution. 
 
Tasks related to visual data
4. Given a location ID, a model and value “k”, returns the most similar k locations based on the corresponding visual descriptors. For each match, also list the overall matching score as well as the 3 image pairs that have the highest similarity contribution. 
5. Given a location ID and value “k”, returns the most similar k locations based on the corresponding visual descriptors. For each match, also list the overall matching score and the individual contributions of the 10 visual models. 


