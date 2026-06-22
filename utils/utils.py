
def obj2dict(obj):
    " Changes from object variables to dictionary"
    a=dict((key, value) for key, value in obj.__dict__.items() 
         if not callable(value) and not key.startswith('__'))
    return a
                
class dict2obj:
    " Dictionary 2 objetc use: a=dict2obj(**dictio) "
    def __init__(self, **entries):
        self.__dict__.update(entries)
