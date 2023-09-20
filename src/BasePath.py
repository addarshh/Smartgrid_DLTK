import os
current_module = __import__(__name__)
base_path = os.path.dirname(current_module.__file__)
base_path = base_path.replace("\\", "/")
base_path = base_path.replace("/src", "/")
