import nbformat
from nbconvert import PythonExporter

# Load the notebook
notebook_path = '/mnt/data/The Job Hunting Crew.ipynb'
with open(notebook_path, 'r') as notebook_file:
    notebook_content = nbformat.read(notebook_file, as_version=4)

# Convert the notebook to a Python script
python_exporter = PythonExporter()
python_script, _ = python_exporter.from_notebook_node(notebook_content)

# Save the Python script
python_script_path = '/mnt/data/The Job Hunting Crew.py'
with open(python_script_path, 'w') as python_file:
    python_file.write(python_script)

python_script_path
