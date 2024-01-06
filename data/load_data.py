
def unpickle(file):
    import pickle
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
