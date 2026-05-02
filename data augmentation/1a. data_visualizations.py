import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects
import pandas as pd
import os

input_data = r'C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\pines_crop_test\indian_pine_array_0_0_0.npy'
data = np.load(input_data)
# create_band_sheet(3, 2)
input_data = r'C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\pines_crop_test_ampshifted\indian_pine_array_0_0_0_ampshifted_1000.npy'
data = np.load(input_data)
# create_band_sheet(3, 2)
# input_data = r'C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\pines_crop_test_rotated\indian_pine_array_0_0_0_rotated_90.npy'
# data = np.load(input_data)
# create_band_sheet(3, 2)

# The data that is being analyzed (.npy file), and the file that the data will be written into 
data = np.load(input_data)

# ---------- Aux Functions (These functions probs won't be called, but they're used for the other functions) ---------- #
file_path = data

def normalize_pixels(array_to_normalize: np.array, maximum_pixel: int, new_max:int):
    '''
    I have to normalize the pixels because its in watts per square meter per steradian (whut) 
    
    Function takes two parameters: 
    array_to_normalize is a numpy array. It is the array with values in watts per square meter per steradian (this function is more or less attatched to the create_band_sheet function)
    maximum_pixel is an integer that is the maximum value in watts per square meter per steradian 
    '''
    
    # Gets the size of the data that will be created so we can iterate all the data
    data_shape = data.shape

    # Creates the array that we will be putting data into once the data is normalized. Creates an array that is the same size as the data but with all 0s
    normalized_array = np.zeros((data.shape[0], data.shape[1]))

    # normalizes the data 
    for number1 in range(0, data_shape[0]):
        for number2 in range(0, data_shape[1]):
            to_put = (array_to_normalize[number1][number2] / maximum_pixel) * new_max

            normalized_array[number1][number2] = to_put

            print(normalized_array[number1][number2])

    return normalized_array

# Specific band for a certain pixel
def pixel_wavelength_information(result=[int, int, int]):
    '''
    Takes a pixel and prints out a particular wavelength for it in the specified band. Takes 3 integers in a list, first two inputs (result[0] and result[1]) 
    correspond to the spot they take up on the image, and the third input (result[2]) corresponds to the wavelength band that you want to use
    '''
    
    # result[0] is the x component and result[1] is the y component and result [2] is the specific wavelength you want to see
    return (data[result[0]][result[1]][result[2]])

# Gets all the wavelength data for a particular pixel
def pixel_information(result=[int, int]):
    '''
    Takes a pixel and prints out all the wavelengths for that particular pixel. result[0] corresponds to x location and result[1] corresponds to y location on the image
    '''
    
    # result[0] is the x component and result[1] is the y component
    return (data[result[0]][result[1]])

def create_a_panda(wavelength: int): 
    '''
    Creates up a panda
    '''

    # Sets the thing to infinity so none of that "..." truncating funny business happens
    np.set_printoptions(threshold=np.inf)
    
    # Gets the shape of the data, only ever tried this on the indian pines dataset but hopefully it'll work for anything 
    data_shape = data.shape

    # Makes an array with data_shape[0] * data_shape[1] pixels to put numbers into 
    my_array = np.zeros((data_shape[0], data_shape[1]))
    
    # This gets the highest watt per square meter per steradian for the normalization function
    maximum_pixel_brilliance = 0
    
    # iterates through the elements in the array and puts it into my_array, units are in watts per square unit per steradian
    for number1 in range(0, data_shape[0]):
        for number2 in range(0, data_shape[1]):
                      
            # Specifies the pixel you want the information for, and the band you want it from
            to_get = [number1, number2, wavelength]
            
            # Passes above information into the pixel_wavelength_information, and it gets the (brilliance?) of a pixel
            to_put = pixel_wavelength_information(to_get)
            
            # To find the maximum brightness pixel, the max is compared
            maximum_pixel_brilliance = max(maximum_pixel_brilliance, to_put)
            
            # The numpy array at [number1][number2] becomes the value that was found in the variable to_put
            my_array[number1][number2] = to_put

    df = pd.DataFrame(my_array)

    print(df)
    
    df.to_csv(f"{wavelength}_panda.csv", index=False)

# ----- Real Functions (this is the stuff that'll get used I think) ----- #

def cook_a_line(wavelength:int, height: int):
    '''
    Gets you a graph for a line at a certain height at a certain wavelength. 
    
    wavelength: Represents the wavelength band you are selecting
    height: I think its from the top down???? so maybe it should be depth??? idk ill figure it out in the future
    '''
    
    # Gets the size of the data that will be created so we can iterate all the data
    data_shape = data.shape
    
    to_graph_array = []
    numbers = []
    
    for pixel_number in range(data_shape[1]):
        first = pixel_information([height, pixel_number])
        pixel_informations = pixel_wavelength_information([height, pixel_number, wavelength])
        
        to_graph_array.append(pixel_informations)
        numbers.append(pixel_number)
        
    plt.title("Band Graph")
    plt.xlabel("Location")
    plt.ylabel("Brilliance")
    plt.plot(numbers, to_graph_array)
    plt.show()

def create_band_sheet(wavelength: int, selection: int, selection_number:int = 255):
    # WORKS PROPERLY!!
    '''
    Creates a big fat 145 * 145 numpy array of all the pixels at a certain band/wavelength, then creates an image using Pillow 
    
    Parameter 1: wavelength integer is the index of the wavelength you want to call (might change this over so you can put in the actual 
    wavelength, but that is a something for another time :D)
    
    Parameter 2: 1 indicates it is normalized to 255, 2 indicates use brilliance values
    '''
    
    # Sets the thing to infinity so none of that "..." truncating funny business happens
    np.set_printoptions(threshold=np.inf)
    
    # Gets the shape of the data, only ever tried this on the indian pines dataset but hopefully it'll work for anything 
    data_shape = data.shape
    
    # this and the "thing += 1" in the nested for loop below counts the number of pixels, should match up to data_shape[0] * data_shape[1], if it doesn't ya done goofed up 
    #thing = 0
    
    # Makes an array with data_shape[0] * data_shape[1] pixels to put numbers into 
    my_array = np.zeros((data_shape[0], data_shape[1]))
    
    # This gets the highest watt per square meter per steradian for the normalization function
    maximum_pixel_brilliance = 0
    
    # iterates through the elements in the array and puts it into my_array, units are in watts per square unit per steradian
    for number1 in range(0, data_shape[0]):
        for number2 in range(0, data_shape[1]):
                      
            # Specifies the pixel you want the information for, and the band you want it from
            to_get = [number1, number2, wavelength]
            
            # Passes above information into the pixel_wavelength_information, and it gets the (brilliance?) of a pixel
            to_put = pixel_wavelength_information(to_get)
            
            # To find the maximum brightness pixel, the max is compared
            maximum_pixel_brilliance = max(maximum_pixel_brilliance, to_put)
            
            # The numpy array at [number1][number2] becomes the value that was found in the variable to_put
            my_array[number1][number2] = to_put

    # Normalizes the array using the normalize_pixels function
    if selection == 1:
        normalized_array = normalize_pixels(my_array, maximum_pixel_brilliance, selection_number)
 
        plt.imshow(normalized_array)

        # img.show()
        plt.show()
    else: 
        plt.imshow(my_array)
        plt.show()

def interactive_3d_pixel_line_display(pixel_x:int, pixel_y:int):
    '''
    
    Gets all the wavelengths for a pixel and graphs it out
    
    pixel_x = x location of pixel 
    pixel_y = y location of pixel 
    
    Same as the pixel_graph function, except made in plotly, which gives more interactivitiy abilities and I can also graph the vertical lines
    
    Uses something called plotly to run, must be run in a interactive window (or jupyter)
    
    Import ipykernel and pip install --upgrade nbformat and hopefully it works
    
    
    '''

    # Gets the brilliance values for all the wavelengths of the pixel location provided
    thing = pixel_information([pixel_x, pixel_y])

    # Shape of the data 
    data_shape = thing.shape

    # Two lists that are appended to, creating the stuff that will be used to graph later
    thing_1 = []
    thing_2 = []

    # Goes over every wavelength and adds its brillaince value to thing_2, the wavelength is added to thing_1
    for thinint in range(data_shape[0]):
        thing_1.append(thinint * metadata[1] + metadata[0])
        thing_2.append(thing[thinint])

    # Makes a panda table with the lists created above to be made into a list later 
    df = pd.DataFrame({"Wavelength": thing_1, "Brilliance": thing_2})

    # Graph created 
    fig = px.line(df, x="Wavelength", y="Brilliance")

    # Iterates over all the minimum wavelengths and puts them onto the graph 
    for minimum_wavelength in spectral_monuments_min:

        # Used to make sure the list is longer than 1, it is is, a rectangle is added
        try:

            # DO NOT DELETE THIS, this triggers an error if its only a single element (variable isn't used that butat's how its supposed to be) 
            items = len(minimum_wavelength)
       
            # Rectangle added to the figure
            fig.add_vrect(x0=minimum_wavelength[0], x1=minimum_wavelength[1], opacity=0.1, line_width=0, fillcolor="red")
            
        # If the length of minimum wavelength gives an error, it is not a list, so it only makes a line 
        except TypeError:
            
            # Line added to the figure 
            fig.add_vline(x=minimum_wavelength, line_width=0.5, line_dash="dash", line_color="red", opacity=0.75)
            
    for maximum_wavelength in spectral_monuments_max:

        # Used to make sure the list is longer than 1, it is is, a rectangle is added
        try:
            
            # DO NOT DELETE THIS, this triggers an error if its only a single element (variable isn't used that butat's how its supposed to be) 
            items = len(maximum_wavelength)
            
            # Rectangle added to the figure
            fig.add_vrect(x0=maximum_wavelength[0], x1=maximum_wavelength[1], opacity=0.1, line_width=0, fillcolor="blue")
            
        # If the length of minimum wavelength gives an error, it is not a list, so it only makes a line 
        except TypeError:
            
            # Line added to the figure 
            fig.add_vline(x=maximum_wavelength, line_width=0.75, line_dash="dash", line_color="blue", opacity=0.75)

    # Figure is displayed 
    fig.show()

def interactive_3d_graph_display(wavelength: int):
    '''
    What does this function do o-O 

    Hmm, I wonder why it only takes in wavelength and nothing else, what magic does it perform :O 
    '''
    supposed_path = f"{wavelength}_panda.csv"

    if not os.path.exists(supposed_path):
        print("hi")
        create_a_panda(wavelength)

    z_data = pd.read_csv(supposed_path)
    # Convert all to floats, exclude non-numeric columns if any
    z = z_data.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)

    print(f"CSV data shape: {z.shape}")

    wavelength_val = 10 * wavelength + 400
    print(f"Plotting wavelength: {wavelength_val}")

    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)

    fig = plotly.graph_objects.Figure(data=[plotly.graph_objects.Surface(z=z, x=x, y=y)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

    fig.update_layout(title=dict(text=f'Wavelength {wavelength_val}'), autosize=False,
                    width=800, height=800,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

def cook_a_line(wavelength:int, height: int):
    '''
    Gets you a graph for a line at a certain height at a certain wavelength. 
    
    wavelength: Represents the wavelength band you are selecting
    height: I think its from the top down???? so maybe it should be depth??? idk ill figure it out in the future
    '''
    
    # Gets the size of the data that will be created so we can iterate all the data
    data_shape = data.shape
    
    to_graph_array = []
    numbers = []
    
    for pixel_number in range(data_shape[1]):
        first = pixel_information([height, pixel_number])
        pixel_informations = pixel_wavelength_information([height, pixel_number, wavelength])
        
        to_graph_array.append(pixel_informations)
        numbers.append(pixel_number)
        
    plt.title("Band Graph")
    plt.xlabel("Location")
    plt.ylabel("Brilliance")
    plt.plot(numbers, to_graph_array)
    plt.show()

create_band_sheet(3, 2)
cook_a_line(5, 3)
# pixel_graph(30, 30)
# interactive_3d_pixel_line_display(15, 15)

interactive_3d_graph_display(3)
