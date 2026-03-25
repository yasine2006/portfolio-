import sys
import os

# Detect project path automatically from this file's location
project_home = os.path.dirname(os.path.abspath(__file__))

if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)

from server_enhanced import app as application
