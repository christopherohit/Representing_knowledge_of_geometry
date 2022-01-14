import logging
import pandas as pd
from vncorenlp import VnCoreNLP
import nltk

import sys

class Extraction_Feature:
    def __init__(self):
        self._model = None
        loaded_model_json = json_file.read()
        self.__model = model_from_json(loaded_model_json)
        self.__model.load_weights("D:\\Workspace\\VisualStudioProj\\WindowsToPythonAI\\python_model\\model.h5")
        self.__model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
        
