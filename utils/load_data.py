import pickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        byte_dict = pickle.load(fo, encoding='bytes')
        
        # Convert the keys from bytes to strings
        string_dict = {key.decode(): value for key, value in byte_dict.items()}

        # Check if the values are byte strings or lists of byte strings, and convert them
        for key, value in string_dict.items():
            if isinstance(value, bytes):
                # Convert byte strings to strings
                string_dict[key] = value.decode()
            elif isinstance(value, list) and all(isinstance(item, bytes) for item in value):
                # Convert lists of byte strings to lists of strings
                string_dict[key] = [item.decode() for item in value]
                
    return string_dict


def load_cifar10():
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to adjust it for Python 3.
            
    Returns
    -------
    X : numpy.ndarray
        Array of training data, reshaped into image dimensions.
    y : numpy.ndarray
        Array of training labels.
    X_test : numpy.ndarray
        Array of test data, reshaped into image dimensions.   
    y_test : numpy.ndarray
        Array of test labels.  
    labels : list
        List of class names.
    """

    # Parse the filepaths for the data batches
    file_paths = ['data/' + file_name for file_name in os.listdir("data") if (file_name.startswith("data") ) ]

    # Load each batch and store the data dictionaries in a list
    dictionaries = [unpickle(file_path) for file_path in file_paths]
    # Load the meta-data for the overall dataset
    meta_data = unpickle("data/batches.meta")

    # Unpack the test data
    file_path = "data/test_batch"
    test_dict = unpickle(file_path)
    X_test = test_dict["data"]
    y_test = test_dict["labels"]  
    
    # Extract the label names and display
    label_names = meta_data["label_names"]
    
    # Concatenate the data from each batch into a single array
    X = np.concatenate([dictionary["data"] for dictionary in dictionaries], axis=0)
    # Concatenate the labels from each batch into a single array
    y = np.concatenate([dictionary["labels"] for dictionary in dictionaries], axis=0)
    
    return X, y, X_test, y_test, label_names