PROVIDED FILES:
decision_tree.py:	python code
report.pdf:		report
wifi_db:	contains two txt files. Each line in the txt files has readings from 7 wifi sources, 
		plus which room the readings are recorded from as the last column. 
		Column are seperated by tabs, lines are seperated by line breaks.
wifi_db/clean_dataset.txt:		clean dataset with correct room labels for each reading.	
wifi_db/noisy_dataset.txt:		noisy dataset with some incorrect room labels for some readings.


BEFORE RUNNING THE CODE:
- make sure you have numpy and mathplotlib installed and loaded in your environment.
- OPTIONAL: Replace the clean_dataset.txt and/or the noisy_dataset.txt with new dataset.
		 Note that the confusion matrix calculation is tailored for 4 rooms, 
		 and will not work with data that have different number of rooms.
		 The code will run with provided dataset if the dataset is not changed.

TO RUN THE CODE:
- open terminal and set working directory to this folder.
- type the following command on the next line:
python3 decision_tree.py

the code will train the decision trees on given datasets, 
then print the resulting trees and evaluation metrics.


You can check report.pdf for details about implementation and results.