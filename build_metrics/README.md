#### File Breakdown 
All of the different files that you can find in this folder and a general overview of what it will do. 
1. ADDING_METRICS.d
Information on how to add new metrics to the metric testing suite. 
2. data_converter.py 
Converts all npy files into mat files (which is what the models are typically expecting to run on). Pretty easy to add compatibility for different files, but I've only seen .npy files so far so I'm not going to do that yet. 
3. data_loader.py
Used to load in the data from the mat files, tests for a bunch of keys to hopefully access the data on the first run. Provides information about the file if it is unable to open the data
4. metrics.py 
Location of the metric code (or a wrapper to the metric code that is stored somewhere else) 
5. run_metrics.py 
Everything in one file (essentially the main.py file), loads in the data, converts it if necessary and stores it into the correct location. 
Performs the different metric calculations on the data that has been created 


### Things to do
1. Test out the pipeline when I get the noisy data files
2. 
