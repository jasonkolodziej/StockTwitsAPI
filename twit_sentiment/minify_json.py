#!/usr/bin/env python3
"""JSON minify program. """

import json # import json library
import sys # import sys library

if __name__ == "__main__":
    """Minify JSON"""
    file_name = 'samples/dataSample_merge.json'
    file_data = open(file_name, "r", 1).read() # store file info in variable
    json_data = json.loads(file_data) # store in json structure
    json_string = json.dumps(json_data, separators=(',', ":")) # Compact JSON structure
    file_name = str(file_name).replace(".json", "") # remove .json from end of file_name string
    new_file_name = "{0}_minify.json".format(file_name)
    open(new_file_name, "w+", 1).write(json_string) # open and write json_string to file