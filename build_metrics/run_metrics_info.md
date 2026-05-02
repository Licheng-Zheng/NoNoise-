#### Purpose
Essentially what I have already typed out, and what I need to implement still bolded. Definitely not the best way to organize everything that I need to do, but it should be a temporary thing to see what I need to do. (I don't have anything bolded currently because I want to make sure what I do have runs with real data that I can get before I start getting into harder to implement stuff)

#### Code Details 
More or less the sequential process that will be run. 

1. main function is what you will be editing in the future to provide all of the paths that will be operated upon.
   1. I can create a gui for the future that allows you find the files for people who don't want to mess around with the code, but that is not something that I want to do right now.
   2. Passes the file paths into the config_mode function that tries to figure out what you want to do. Currently, allows for two modes that checks if you want to compare the paths provided (so you can only pass in one set of data at a time) and providing a path to the folder containing everything (you can do things a lot faster this way)
   3. Not sure if it works yet because I've never tried it, but it should be too hard of a fix if it doesn't work.

2. Packages all the paths for the data and puts all of the different metrics that we want to run into a dictionary that all the information will be stored within
   1. metric_context contains all of the context that is passed into the different metrics. For example, the max_val used by the PSNR, the device used by SSIM (what device will the metric be run on). It is passed in as a kwargs (or was it args idk), and the metric function takes the information that it needs from the dictionary, and ignores the stuff that it doesn't.

3. Loads all the data hopefully everything loads in properly. Prints out different error messages if it doesn't.

4. Runs all the different metrics and puts it into the dictionary created earlier.
   1. Puts everything into a csv file that is saved and returned. This is what will be used as the standard output format in the future, because you can refer to it easily and don't need to save it yourself.
   2. Also prints it out into the console, but it might be a bit ugly as we add more metrics, so the CSV is probably the best way to go forward