""" Functions for converting from DOS to UNIX line endings
"""

import sys, re, os

def dos2unix(file):
    "Replace CRLF with LF in argument files.  Print names of changed files."    
    if os.path.isdir(file):
        print file, "Directory!"
        return
        
    data = open(file, "rb").read()
    if '\0' in data:
        print file, "Binary!"
        return
        
    newdata = re.sub("\r\n", "\n", data)
    if newdata != data:
        print file
        f = open(file, "wb")
        f.write(newdata)
        f.close()
    else:
        print file, 'ok'    

def dos2unix_one_dir(args,dir_name,file_names):
    for file in file_names:
        full_path = os.path.join(dir_name,file)
        dos2unix(full_path)
    
def dos2unix_dir(dir_name):
    os.path.walk(dir_name,dos2unix_one_dir,[])
    
if __name__ == "__main__":
    import sys
    dos2unix_dir(sys.argv[1])
