
# coding: utf-8

# In[2]:

import codecs


# In[3]:

def open_file(file_name, mode):
    """Open a file."""
    try:
        the_file = codecs.open(file_name, mode, "utf-8") 
    except(IOError), e:
        print "Unable to open the file", file_name, "Ending program.\n", e
        raw_input("\n\nPress the enter key to exit.")
        sys.exit()
    else:
        return the_file

def open_file2(file_name, mode):
    """Open a file."""
    try:
        the_file = open(file_name, mode) 
    except(IOError), e:
        print "Unable to open the file", file_name, "Ending program.\n", e
        raw_input("\n\nPress the enter key to exit.")
        sys.exit()
    else:
        return the_file

# In[ ]:



