def _init():  # initialization
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    #global variable
    _global_dict[key] = value

def get_value(key):
    #get a global variable 
    try:
        return _global_dict[key]
    except:
        print('get'+key+'false\r\n')