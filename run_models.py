import os 
import importlib.util

# Figure out how to get it working with actual models (.keras, .pth, etc (that's all I know)) 

def models_on_dataset(dataset): 
    # Where the models can be found with their information (relative path) 
    model_folder_path = r"model"
    dataset_folder_path = r"database"
    output_folder_path = r"processed"

    required_file_type = "mat_"

    # Load the dataset to be operated on (the noisy one to be cleaned up) 
    noisy_dataset_path = os.path.join(dataset_folder_path, dataset, "noisy.mat")
    
    if noisy_dataset_path.lower().endswith(required_file_type):
        return f"Requires file of type {required_file_type}"

    # Check if the output director exists, if not, create it 
    folders_in_processed = os.listdir(output_folder_path)
    
    # os.makedirs(folder_path, exist_ok=True) is also doable, it doesn't create anything if it already 
    # exists, but the if statement makes it easier to read (also did not know that existed) 
    if dataset not in folders_in_processed: 
        os.makedirs(os.path.join(output_folder_path, dataset))
    
    processed_output_folder_path = os.path.join(output_folder_path, dataset)               

    # All the different models that are available (the dataset is passed through all of them) 
    model_directories = os.listdir(model_folder_path)

    for model in model_directories: 
        # Include different ways of running models later (pth files and such, .py is just a dummy model) 
        model_file = os.path.join(model_folder_path, model, "model.py")
        
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Call the 'model' function if it exists, I need to create functionality for other models in the future,
        # this is just a proof of concept for now because I can't get the models working -_- 
        if hasattr(module, 'model'):
            result = module.model(
                input_mat_path=noisy_dataset_path,
                input_var="data", # Dummy variable name, change later
                output_mat_path=os.path.join(processed_output_folder_path, f"{model}.mat"),
                output_var="data" # Dummy variable name, change later
            )
        else:
            print("NOT FOUND IT!!") 

models_on_dataset("ksc512")



    
