import os
import json

# Define the names of files and folders to be ignored
ignored_files = [".env", ".gitignore", "database.db", ".DS_Store"]
ignored_folders = ["__prompt", "__info", "build", "__json_files", "__pycache__", ".git", "temp_project", "ai"]
CROP_NUM_CHARS = 60

# Define a file prefix to be ignored
file_prefix_to_ignore = "__"

def ensure_directory_exists(path):
    """
    Ensures that the specified directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def crop_strings(data):
    """
    Recursively crop all string values in a dictionary to a maximum of CROP_NUM_CHARS characters.
    """
    if isinstance(data, str):
        return data[:CROP_NUM_CHARS]
    elif isinstance(data, dict):
        return {key: crop_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [crop_strings(item) for item in data]
    else:
        return data

def load_memory_json():
    """
    Load memory.json if it exists, or return an empty dictionary.
    """
    if os.path.exists("memory.json"):
        with open("memory.json", "r", encoding="utf-8") as memory_file:
            return json.load(memory_file)
    else:
        return {}

def save_memory_json(memory_data):
    """
    Save memory.json with cropped string values.
    """
    with open("memory.json", "w", encoding="utf-8") as memory_file:
        json.dump(crop_strings(memory_data), memory_file, indent=4)

def init_markdown_content():
    """
    Initialize the markdown content with the default header.
    """
    initial_information = "## initial information:\n" + \
        "please check all provided data\n\n" + \
        "- I am providing you with the project description\n" + \
        "- I am providing you with the module json file structure\n" + \
        "- I am providing you with initial implementations for a few classes\n\n\n"
        
    initial_instructions = "## instructions:\n" + \
        "- please study the provided information and data very well. respond only with YES after the review.\n\n"
        #"- please fix saving data to memory in the InputStepProcessor class\n" + \
        #"- update the class to properly load the outputkeys and use them as memory key.\n" + \
        #"- it is a multi level output key\n\n\n"
        
    initial_markdown_content = initial_information + initial_instructions

    return initial_markdown_content

def generate_prompt_context(project_path):
    """
    Generates prompt context based on the content of all files in the given project path.
    """
    if not os.path.exists(project_path):
        return "Project path does not exist.", "", ""

    markdown_files = []
    json_files = []
    py_files = []
    other_files = []
    file_structure = {}

    for root, dirs, files in os.walk(project_path):
        # Remove ignored files and folders from the lists
        files[:] = [f for f in files if f not in ignored_files and not f.startswith(file_prefix_to_ignore)]
        dirs[:] = [d for d in dirs if d not in ignored_folders]

        for name in files:
            
            file_path = os.path.join(root, name)
            relative_file_path = os.path.relpath(file_path, project_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                
                if name.endswith(".md"):
                    markdown_files.append((relative_file_path, file_content))
                elif name.endswith(".json"):
                    if name == "memory.json":
                        # load memory.json if it exists
                        memory_data = json.loads(file_content)
                        cropped_data = crop_strings(memory_data)
                        json_files.append((relative_file_path, json.dumps(cropped_data, indent=4)))
                    else:
                        json_files.append((relative_file_path, file_content))
                elif name.endswith(".py"):
                    py_files.append((relative_file_path, file_content))
                else:
                    other_files.append((relative_file_path, file_content))

                folder_path = os.path.dirname(relative_file_path)
                if folder_path not in file_structure:
                    file_structure[folder_path] = []
                file_structure[folder_path].append(name)
            except Exception as e:
                markdown_files.append((relative_file_path, f"Error reading file: {e}"))

    # Concatenate the lists in the desired order
    prioritized_files = markdown_files + json_files + py_files + other_files

    # Generate content for markdown, json, and file structure
    markdown_content = init_markdown_content()
    json_content = {}
    for relative_file_path, file_content in prioritized_files:
        markdown_content += f"## {relative_file_path}:\n\n{file_content}\n\n"
        json_content[relative_file_path] = file_content

    file_structure_json = json.dumps(file_structure, indent=4)
    markdown_content += "## File and Folder Structure:\n\n" + file_structure_json

    return markdown_content, json.dumps(json_content, indent=4), file_structure_json

# Load memory.json if it exists
memory_data = load_memory_json()

# Generate prompt context
project_path = "./"
markdown_content, json_content, file_structure_json = generate_prompt_context(project_path)

# Save the modified memory.json
# save_memory_json(memory_data)

# Ensure the output directory exists
output_dir = "./__prompt"
ensure_directory_exists(output_dir)

# Output to files
md_file_path = f"{output_dir}/prompt_context.md"
with open(md_file_path, "w") as md_file:
    md_file.write(markdown_content)
    print (f"Markdown file written to {md_file_path}")

with open(f"{output_dir}/prompt_context.json", "w") as json_file:
    json_file.write(json_content)

with open(f"{output_dir}/file_structure.json", "w") as fs_file:
    fs_file.write(file_structure_json)
