## initial information:
please check all provided data

- I am providing you with the project description
- I am providing you with the module json file structure
- I am providing you with initial implementations for a few classes


## instructions:
- please study the provided information and data very well. respond only with YES after the review.

## README.md:

# maldinio_ai

## dist/maldinio_ai-0.1.1-py3-none-any.whl:

Error reading file: 'utf-8' codec can't decode byte 0xe0 in position 10: invalid continuation byte

## dist/maldinio_ai-0.1.1.tar.gz:

Error reading file: 'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte

## setup.py:

from setuptools import setup, find_packages

setup(
    name='maldinio_ai',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        "tiktoken>=0.4.0",
        "openai>=1.3.7",
    ],
    description='A utility package for AI prompt management and prompt processing.',
    author='Mehdi Nabhani',
    author_email='mehdi@nabhani.de',
    keywords=['AI', 'NLP', 'LLM', 'Prompt Management'],

)


## tools/create_project_folder.py:

import os
import json
from datetime import datetime
from ai import ModuleMemory, NLPProcessor

class CreateProjectFolder:
    def __init__(self, memory: ModuleMemory):
        self.main_key = "project"
        self.key = "files"
        self.sub_key = "project_folder"
        self.memory = memory
        self.root_folder = "temp_project"
        self.project_name = "project_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_path = os.path.join(self.root_folder, self.project_name)
        self.full_path_prompts = os.path.join(self.root_folder, self.project_name, "prompts")
        self.full_path_responses = os.path.join(self.root_folder, self.project_name, "responses")
        self.full_path_output = os.path.join(self.root_folder, self.project_name, "output")

    def get_key(self):
        return self.key
    
    def get_main_key(self):
        return self.main_key
    
    def get_sub_key(self):
        return self.sub_key

    def execute(self):
        self.create_project_directory()

    def create_project_directory(self):
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
            print(f"Created project directory: {self.full_path}")
        else:
            print(f"Project directory already exists: {self.full_path}")
                
        if not os.path.exists(self.full_path_prompts):
            os.makedirs(self.full_path_prompts)
            print(f"Created prompts directory: {self.full_path_prompts}")
        else:
            print(f"Prompts directory already exists: {self.full_path_prompts}")
            
        if not os.path.exists(self.full_path_responses):
            os.makedirs(self.full_path_responses)
            print(f"Created responses directory: {self.full_path_responses}")
        else:
            print(f"Responses directory already exists: {self.full_path_responses}")
            
        if not os.path.exists(self.full_path_output):
            os.makedirs(self.full_path_output)
            print(f"Created output directory: {self.full_path_output}")
        else:
            print(f"Output directory already exists: {self.full_path_output}")
            
        self.memory.create([self.main_key, self.key, self.sub_key], self.full_path)
        self.memory.create([self.main_key, self.key, "prompt_folder"], self.full_path_prompts)
        self.memory.create([self.main_key, self.key, "response_folder"], self.full_path_responses)
        self.memory.create([self.main_key, self.key, "output_folder"], self.full_path_output)


## tools/load_project.py:

# modules/load_project.py

import os
import json
from ai import ModuleMemory, NLPProcessor

class LoadProject:
    def __init__(self, memory: ModuleMemory):
        self.main_key = "project"
        self.key = "initial_project_details"
        self.memory = memory
        
    def get_key(self):
        return self.key
    
    def get_main_subkey(self):
        return self.main_subkey
    
    def execute(self):
        # Get the current working directory
        current_directory = os.getcwd()

        # Filename you want to join with the current directory
        filename = "project.json"

        # Join the current directory with the filename
        project_file_path = os.path.join(current_directory, filename)
        
        print ("loading project:", project_file_path)

        self.load_project(project_file_path)
        # self.enhance_project()

    def load_project(self, project_file):
        """
        Load project details from a JSON file and store them in memory.
        """
        try:
            with open(project_file, 'r') as file:
                project_data = json.load(file)
                self.memory.create([self.main_key, self.key], project_data)
                print(f"Project '{project_data['name']}' loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File '{project_file}' not found.")
        except json.JSONDecodeError:
            print("Error: JSON decoding error.")
        except Exception as e:
            print(f"An error occurred: {e}")



## utils/verification_utils.py:

import json
from utils.json_utils import extract_json_from_message, extract_json_string_from_message


def validate_json_structure_noarray(json_data, expected_structure):
    for key, value_type in expected_structure.items():
        print (key, value_type)
        if key not in json_data or not isinstance(json_data[key], value_type):
            return False
    return True

def validate_json_structure(json_data, expected_structure):
    for key, value_type in expected_structure.items():
        print ('key, value_type', key, value_type)
        
        
        if key not in json_data:
            print ('key not in json_data')
            
            return False
        if isinstance(value_type, list):
            print ('value_type is list')
            if not isinstance(json_data[key], list):
                print ('json_data[key] is not list')
                return False
            # Validate each item in the array
            for item in json_data[key]:
                print ('item', item)
                if not isinstance(item, value_type[0]):
                    print ('item is not value_type[0]')
                    return False
        else:
            print ('value_type is not list')
            if not isinstance(json_data[key], value_type):
                print ('json_data[key] is not value_type')
                return False
    return True


def validate_data(data):
    # Add your specific validation logic here
    return True

def validate_subtasks(subtasks):
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            return False
        expected_keys = ["id", "name", "description", "status"]
        for key in expected_keys:
            if key not in subtask or not isinstance(subtask[key], str):
                return False
    return True

## utils/json_utils.py:

import re
import json

def fix_json(json_data):
    try:
        # Attempt to directly parse the JSON first
        return json.loads(json_data)
    except json.JSONDecodeError:
        try:
            # Attempt to fix single quotes and re-parse
            fixed_json = json_data.replace("'", '"')
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            # Handle other JSON errors
            return None

def verify_json(json_data, structure):
    
    debug = False

    if debug == True:
        print ("json_data: ", json_data)
        print ("structure: ", structure)
        print ("--------------------------------")
        print ()
        
        data = json.loads(json_data)

        comparison_result = compare_structure(data, structure)

    data = fix_json(json_data)
    if data is not None:
        comparison_result = compare_structure(data, structure)
        return comparison_result
    else:
        # Handle other JSON errors
        return False
        
def compare_structure(data, structure):
    if isinstance(structure, type):
        return isinstance(data, structure)

    if isinstance(structure, dict):
        for key, value_structure in structure.items():
            if key not in data or not compare_structure(data[key], value_structure):
                return False

    elif isinstance(structure, list):
        # Handling list of primitives separately
        if isinstance(structure[0], type):
            return all(isinstance(item, structure[0]) for item in data)
        else:
            if not all(isinstance(item, type(structure[0])) for item in data):
                return False
            for item in data:
                if not compare_structure(item, structure[0]):
                    return False

    return True



def generate_string_from_json(json_obj):
    def process_value(value):
        if isinstance(value, dict):
            return generate_string_from_json(value)
        elif isinstance(value, list):
            if value:
                element = process_value(value[0])
                return f"[{element}]"
            else:
                return "[]"
        else:
            return str(value)


    string_representation = "{"
    for key, value in json_obj.items():
        processed_value = process_value(value)
        string_representation += f'"{key}": {processed_value}, '
    string_representation = string_representation.rstrip(", ")
    string_representation += "}"
    return string_representation

# Example usage
expected_structure = {
    "nlp_task": str,
    "task_description": str,
    "nested_array": [str],
    "nested_object": {
        "property1": int,
        "property2": float
    }
}




def convert_json_to_python_object(json_dict):
    # Define a dictionary to store the converted Python object structure
    python_object = {}

    # Define a mapping from type string to Python type
    types_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "null": type(None),
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "frozenset": frozenset,
        "complex": complex,
        "bytes": bytes,
        "bytearray": bytearray
    }

    # Iterate over each key-value pair in the JSON dictionary
    for key, value in json_dict.items():
        # If the value is a list, recursively call this function for each element
        if isinstance(value, list):
            python_object[key] = [convert_json_to_python_object(v) if isinstance(v, dict) else types_mapping.get(v, v) for v in value]
        # If the value is a nested dictionary, recursively call this function
        elif isinstance(value, dict):
            python_object[key] = convert_json_to_python_object(value)
        # If the value is a type string, convert it to the corresponding Python type
        else:
            python_object[key] = types_mapping.get(value, value)

    return python_object



# Example usage:
obj = {
    "step_number": int,
    "step_description": str,
    "name": str,
    "description": str,
    "tasks": [
        {
            "task_name": str,
            "task_type": str,
            "tree_of_thought": bool,
            "quality_check": bool,
            "input_values": [str],
            "output_values": [str],
            "prompt": str,
            "expected_structure": {
                "functionalities": [str]
            },
            "replacements": dict
        }
    ]
}


def generate_expected_structure_string(expected_structure):
    expected_structure_object = convert_json_to_python_object(expected_structure)
    expected_structure_object_string = generate_string_from_json(expected_structure)
    return expected_structure_object_string



def convert_python_object_to_json(obj):
    # Helper function to convert Python types to serializable types
    def convert_type(value):
        if isinstance(value, type):
            return value.__name__
        return value

    # Recursively convert the object to JSON-compatible types
    def convert(obj):
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(value) for value in obj]
        else:
            return convert_type(obj)

    # Convert the object to JSON representation
    json_representation = json.dumps(convert(obj), indent=4)
    return json_representation


def extract_json_from_message(message):
    start_token = "{"
    end_token = "}"

    # Find the start and end indices of the JSON object within the message
    start_index = message.find(start_token)
    end_index = message.rfind(end_token)

    if start_index == -1 or end_index == -1:
        return "JSON object not found in the message"

    # Extract the JSON object from the message
    json_string = message[start_index:end_index + len(end_token)]

    # Parse the JSON string into a Python object
    json_data = json.loads(json_string)

    return json_data


def extract_json_string_from_message(message):
    start_token = "{"
    end_token = "}"

    # Find the start and end indices of the JSON object within the message
    start_index = message.find(start_token)
    end_index = message.rfind(end_token)

    if start_index == -1 or end_index == -1:
        return "JSON object not found in the message"

    # Extract the JSON object from the message
    json_string = message[start_index:end_index + len(end_token)]

    return json_string


def cleanup_json_response(response):
    
    if response.startswith("```json"):
        # Use regular expressions to extract the entire JSON block
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        
        if match:
            json_block = match.group(1)
            try:
                # Parse the JSON string
                json_data = json.loads(json_block)
                return json_block

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                # Attempt to fix by adding a closing bracket
                fixed_json_block = json_block + "}"
                try:
                    # Parse the fixed JSON string
                    json_data = json.loads(fixed_json_block)
                    return fixed_json_block
                except json.JSONDecodeError:
                    print("Failed to fix JSON.")
        else:
            print("JSON data not found in the input string.")

    return response

## utils/helpers.py:

import os
import markdown
import json

def fill_gaps_with_underscore(string):
    # Split the string by spaces
    words = string.split()

    # Create a new list to store the modified words
    modified_words = []

    # Iterate over each word
    for word in words:
        # If the word has gaps (multiple consecutive underscores), replace them with a single underscore
        modified_word = word.replace('_', ' ')
        modified_word = modified_word.replace(' ', '_')

        # Add the modified word to the list
        modified_words.append(modified_word)

    # Join the modified words back into a string with spaces
    filled_string = ' '.join(modified_words)

    return filled_string

## prompt/prompt_context.py:

class PromptContext:
    def __init__(self, context_dict=None, **kwargs):
        if context_dict is None:
            context_dict = kwargs

        self.role = context_dict.get('role', '')
        self.prefix = context_dict.get('prefix', '')
        self.suffix = context_dict.get('suffix', '')
        self.list_item = context_dict.get('list_item', {})
        self.context_items = context_dict.get('context_items', {})
        self.context = context_dict.get('context', [])
        self.questions = context_dict.get('questions', [])
        self.examples = context_dict.get('examples', [])
        self.instructions = context_dict.get('instructions', [])
        self.query = context_dict.get('query', '')
        self.simple_prompt = context_dict.get('simplePrompt', '')
        self.response_format = context_dict.get('response_format', '')
        self.response_structure = context_dict.get('response_structure', '')

    def clean_context(self):
        """
        Reset all attributes of the instance.
        """
        for key in vars(self):
            setattr(self, key, None)

    def update_context(self, update_dict):
        """
        Update the context attributes based on the provided dictionary.
        Add new attributes if they do not exist.
        """
        self.clean_context()
        for key, value in update_dict.items():
            setattr(self, key, value)  # This will update or add a new attribute

    def add_attribute(self, key, value):
        """
        Add a new attribute to the instance.
        """
        setattr(self, key, value)
        
    def get_attribute(self, key):
        """
        Get the value of an attribute.
        """
        return getattr(self, key)

    def print_attributes(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def find_unknown_attributes(self, known_attrs):
        """
        Returns a list of attribute names of the instance that are not in the known_attrs list.

        :return: A list of unknown attribute names.
        """
        preset_attrs = ['prefix', 'suffix', 'list_item', 'simple_prompt', 'query', 'context_items', 'questions', 'examples',
                       'context', 'instructions', 'response_format', 'response_structure'] if not known_attrs else known_attrs
        
        return [attr for attr in self.__dict__ if attr not in preset_attrs]

    def print_context(self):
        # Print all attributes of the instance
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"- {sub_key}: {sub_value}")
            elif isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"- {item}")
            else:
                print(f"{key}: {value}")

    def print_context(self):
        print("Role:", self.role)
        print("Prefix:", self.prefix)
        print("Suffix:", self.suffix)
        print("List Item:")
        for key, value in self.list_item.items():
            print(f"- {key}: {value}")
        print("Context Items:")
        for key, value in self.context_items.items():
            print(f"- {key}: {value}")
        print("Context:")
        for item in self.context:
            print(f"- {item}")
        print("Questionaire:")
        for question in self.questions:
            print(f"- {question}")
        print("Examples:")
        for example in self.examples:
            print(f"- {example}")            
        print("Instructions:")
        for instruction in self.instructions:
            print(f"- {instruction}")
        print("Query:", self.query)
        print("Simple Prompt:", self.simple_prompt)
        print("Response Format:", self.response_format)
        print("Response Structure:", self.response_structure)


## prompt/prompt_generator.py:

# from agents.project_items import FileList, File, FunctionalityList, Functionality, AIList
from .prompt_context import PromptContext

def convert_string(string):
    words = string.split('_')
    converted_words = [word.capitalize() for word in words]
    converted_string = ' '.join(converted_words)
    return converted_string


class PromptGenerator:
    def __init__(self, context: PromptContext = None, input_keys_data = None):
        if context is not None:
            self.set_context(context)
        self.input_keys_data = input_keys_data

    def set_context(self, context):
        self.promptcontext = context
        self.prefix = context.prefix or ""
        self.suffix = context.suffix or ""
        self.list_item = context.list_item or {}
        self.prompt = context.simple_prompt or ""
        self.context_items = context.context_items or []
        self.questions = context.questions or []
        self.examples = context.examples or []
        self.context = context.context or []
        self.instructions = context.instructions or []
        self.query = context.query or ""
        self.response_format = context.response_format or ""
        self.response_structure = context.response_structure or ""

    def generate_prompt(self):
        # Handle additional dynamic attributes
        known_attrs = ['role', 'prefix', 'suffix', 'list_item', 'simple_prompt', 'query', 'context_items', 'questions', 'examples',
                       'context', 'instructions', 'response_format', 'response_structure']
        # additional_attrs = [attr for attr in additional_attrs if not callable(getattr(self.context, attr))]
        additional_attrs = self.promptcontext.find_unknown_attributes(known_attrs)

        prompt = ""
        
        if self.prefix != "" and self.prefix is not None:
            prompt = f"{self.prefix}\n\n"
        
        prompt += f"## Prompt:\n{self.prompt}."
        
        if self.response_format != "" and self.response_format is not None:
            prompt += " " + f"Please provide your response in {self.response_format} format"
            prompt += " " + f"and in the requested structure which is shown below."
            prompt += "\n\n"
        else:
            prompt += "\n\n"
        

        if self.query != "" and self.query is not None:
            prompt += f"## Query:\n{self.query}\n\n"
        
        if self.list_item and self.list_item is not None:
            prompt += "## Current List Item:\n"
            for key, value in self.list_item.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        if additional_attrs and additional_attrs is not None:
            prompt += "## Additional Context:\n"
            for attr in additional_attrs:
                value = getattr(self.promptcontext, attr)
                prompt += f"- {attr}: {value}\n"
            prompt += "\n"

        if self.context or self.context_items or self.input_keys_data is not None:
            prompt += "## Context:\n"
            
            if self.context and self.context is not None:
                for context_item in self.context:
                    prompt += f"- {context_item}\n"
                    
            if self.context_items and self.context_items is not None:
                for context_item in self.context_items.items():
                    key , context = context_item
                    # print()
                    # print ("checking key: ", key)
                    # print (context)
                    # print (type(context))
                    # print (isinstance(context, FileList))
                    # print (isinstance(context, FunctionalityList))
                    # print (isinstance(context, Functionality))
                    # print (isinstance(context, File))
                    # print (isinstance(context, str))
                    
                    ## if isinstance(context, FileList):
                    ##     file_list_prompt = context.generate_file_list_context()
                    ##     prompt += file_list_prompt + "\n"
                    ## elif isinstance(context, FunctionalityList):
                    ##     functionality_list_prompt = context.generate_functionality_context()
                    ##     prompt += functionality_list_prompt + "\n"
                    ## elif isinstance(context, AIList):
                    ##     ailist_prompt = context.generate_context()
                    ##     prompt += ailist_prompt + "\n"
                    ## else:
                    ##     key, context = context_item
                    ##     prompt += f"- {key}: {context}\n"
                    
                    prompt += f"- {key}: {context}\n"
            
            if self.input_keys_data:
                for key, value in self.input_keys_data.items():
                    prompt += f"- {key}: {value}\n"

            prompt += "\n"

        if self.instructions and self.instructions is not None:
            prompt += "## Instructions:\n"
            for idx, instruction in enumerate(self.instructions, start=1):
                prompt += f"{idx}. {instruction}\n"
            prompt += "\n"

        if self.questions and self.questions is not None:
            prompt += "## Questionaire:\n"
            for idx, question in enumerate(self.questions, start=1):
                prompt += f"{idx}. {question}\n"
            prompt += "\n"

        if self.examples and self.examples is not None:
            prompt += "## Examples:\n"
            for idx, example in enumerate(self.examples, start=1):
                prompt += f"{idx}. {example}\n"
            prompt += "\n"

        if self.response_format != "" and self.response_format is not None:
            prompt += "## Response Format:\n"
            prompt += f"Remember to provide your response in the format: {self.response_format}\n\n"

        if self.response_structure != "" and self.response_structure is not None:
            prompt += "## Expected Structure:\n"
            prompt += f"{self.response_structure}\n\n"

        prompt += f"{self.suffix}\n"
        
        return prompt



    def set_prompt_prefix(self, prefix = ""):
        
        if prefix == "":
            prompt_prefix = f"""## Project:\nPlease complete below instructions to complete the task. Provide your response in the given response format and in the expected structure.\n\n"""
        else:
            prompt_prefix = prefix

        self.prompt_prefix = prompt_prefix
        
        return prompt_prefix


    def set_prompt_instructions(self, instructions = None):
        
        prompt_text = ""
                
        if instructions is not None:
            prompt_text += f"\n\n## Instructions:\n"
            for instruction in instructions:
                prompt_text += "- " + instruction + "\n"

        self.prompt_body = prompt_text
        
        return prompt_text
    
    
    def set_prompt_body(self, prompt = "", prompt_body_items = None):
        prompt_text = ""
        
        if prompt != "":
            prompt_text = f"""## Task:\n{prompt}\n"""
        else:
            prompt_text = prompt
        
        if prompt_body_items is not None:
            for placeholder, replacement in prompt_body_items.items():
                
                placeholder_title = convert_string(placeholder)

                prompt_text += f"\n\n## {placeholder_title}:\n{{{{{placeholder}}}}}\n"
                placeholder = "{{" + placeholder + "}}"
                prompt_text = prompt_text.replace(placeholder, replacement)


        self.prompt_body = prompt_text
        
        return prompt_text

        

    def set_prompt_suffix(self, response_format = "json", expected_structure = "{ \"key\": \"value\"}"):

        prompt_suffix = f"\n## Response Format:\nRemember to provide your response in minified {response_format} format.In order to save tokens also avoid line breaks in the minified response. Do not add any further comments for automatic processing of the response.\n"

        if expected_structure and expected_structure != "{}":
            prompt_suffix += f"\n## Expected Structure:\n{expected_structure}\n"

        self.prompt_suffix = prompt_suffix
        
        return prompt_suffix
        
    def prepare_prompt_builder(self):
        self.set_prompt_prefix()
        self.set_prompt_body()
        self.set_prompt_suffix()
        self.set_prompt_instructions()
        self.use_prompt_builder = True

## prompt/response_processor.py:

class ResponseProcessor:
    def __init__(self, response_format):
        self.format = response_format

    def process_response(self, response):
        # Implementation to process the NLP response into a structured JSON output
        # Placeholder for response processing logic
        print(f"Processing response: {response}")
        return response


## memory_management/memory_manager.py:

import json

class ModuleMemory:
    def __init__(self):
        self.memory_store = {}

    def _navigate_to_node(self, path_list, create_missing=False):
        """
        Navigate to the node specified by the path list.
        If create_missing is True, missing nodes along the path will be created.
        """
        current_node = self.memory_store
        for key in path_list[:-1]:
            if key not in current_node:
                if create_missing:
                    current_node[key] = {}
                else:
                    raise KeyError(f"Path '{' > '.join(path_list)}' does not exist.")
            current_node = current_node[key]
        return current_node, path_list[-1]

    def save_response(self, response):
        """
        Save the response from OpenAI to memory.
        """
        self.memory_store['response'] = response
        
    def get_response(self):
        """
        Get the response from memory.
        """
        return self.memory_store.get('response')

    def create(self, path_list, value):
        """
        Create a new entry in memory at the specified path.
        """
        node, key = self._navigate_to_node(path_list, create_missing=True)
        if key in node:
            raise KeyError(f"Key '{key}' already exists at path '{' > '.join(path_list)}'.")
        node[key] = value

    def read(self, path_list):
        """
        Read an entry from memory at the specified path.
        """
        
        try:
            node, key = self._navigate_to_node(path_list)
        except KeyError:
            return None
        
        return node.get(key)

    def update(self, path_list, value):
        """
        Update an existing entry in memory at the specified path.
        """
        node, key = self._navigate_to_node(path_list)
        if key not in node:
            raise KeyError(f"Key '{key}' not found at path '{' > '.join(path_list)}'.")
        node[key] = value
        
    def create_or_update(self, path_list, value):
        """
        Create a new entry in memory at the specified path if it doesn't exist,
        otherwise update the existing entry.
        """
        node, key = self._navigate_to_node(path_list, create_missing=True)
        node[key] = value

    def delete(self, path_list):
        """
        Delete an entry from memory at the specified path.
        """
        node, key = self._navigate_to_node(path_list)
        if key in node:
            del node[key]


    def exists(self, key):
        """
        Check if a key exists in memory.
        """
        return key in self.memory_store

    def get_all_keys(self):
        """
        Get all keys in memory.
        """
        return list(self.memory_store.keys())

    def save_to_file(self, file_path):
        """Saves the current state of memory to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(self.memory_store, file, indent=4, sort_keys=True)

    def load_from_file(self, file_path):
        """Loads the current state of memory from a JSON file."""
        with open(file_path, 'r') as file:
            self.memory_store = json.load(file)

## nlp/nlp_processor.py:

import json
import os
from datetime import datetime
from .nlp_client import NLPClient
from ai.utils import extract_json_from_message, extract_json_string_from_message, cleanup_json_response
from ai.utils import fill_gaps_with_underscore, verify_json
from typing import List
from ai.memory_management import ModuleMemory

class NLPProcessor:
    def __init__(self, memory: ModuleMemory = None, nlp_client: NLPClient = None, project_path=None, prompt_path=None, response_path=None):
        self.memory = memory
        self.nlp_client = nlp_client if nlp_client else NLPClient()
        self.project_path = self.memory.read(["project", "files", "project_folder"])
        self.prompt_path = self.memory.read(["project", "files", "prompt_folder"])
        self.response_path = self.memory.read(["project", "files", "response_folder"])
        self.output_path = self.memory.read(["project", "files", "output_folder"])
        self.counter = 0

    def set_project_path(self, project_path):
        self.project_path = project_path

    def set_prompt_path(self, prompt_path):
        self.prompt_path = prompt_path

    def set_response_path(self, response_path):
        self.response_path = response_path

    def process(self, prompt, context):
        """Process the query using the NLP system."""

        role = context.role or "GPT Manager"
        prompt = prompt
        response_format = context.response_format or "json"
        response_structure = context.response_structure or """{ "data" : { "response" : "str" }}"""

        response = self.get_response(role, prompt, response_format, response_structure, "", True)
        #### response = self.nlp_client.process(query)

        content = self.process_response(response)

        return content

    def save_item(self, role, toggle, file_suffix, item):
        # Save item to a text file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{toggle}_{self.counter}_{role}{file_suffix}_{timestamp}.md"

        if toggle == "Prompt":
            save_folder = self.prompt_path
        elif toggle == "Response":
            save_folder = self.response_path
        else:
            save_folder = self.project_path
            
        print ("save_folder: ", save_folder)
        print ("filename: ", filename)
        file_path = os.path.join(save_folder, filename)

        with open(file_path, "w") as file:
            file.write(f"### {toggle} (time: {timestamp})\n\n")
            file.write(item)
            

    def get_verified_response_single(self, role, prompt: str, response_format, response_structure) -> str:
        retries = 10
        
        output_path = self.memory.read(["project", "files", "output_folder"])
        
        while retries > 0:

            # Call the GPT-4 model for each prompt and get the response
            original_response = self.nlp_client.process(prompt, role)
            
            response = cleanup_json_response(original_response)
            
            ##### response_structure = response_structure.replace('bool', 'True')

            # Save failed response as a .md file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            file_name = "__response_collection_" + timestamp + ".md"
            output_filename = os.path.join(self.output_path, file_name)
            
            original_file_name = "__response_original_collection_" + timestamp + ".md"
            original_output_filename = os.path.join(self.output_path, original_file_name)
            
            try :
                with open(original_output_filename, "w") as f:
                    f.write(original_response)
                with open(output_filename, "w") as f:
                    f.write(response)
            except:
                print ()
                print ()
                print ("--------------------------------")
                print ("failed to write response to file")
                print ("--------------------------------")
                print ()
                print ()
                print ("original_response: ", original_response)
                print ("--------------------------------")
                print ()
                print ()
                print ("response: ", response)
                print ("--------------------------------")
                print ()
                print ()
                pass
            
            if response_format == "markdown":
                return response
            
            # Verify the response against the JSON structure
            if verify_json(response, response_structure):
                return response
            
            print ("response verification failed, retries left: ", retries)
            # input ("press enter to continue")

            retries -= 1
        
        self.safe_error_prompt(prompt, role, response, response_structure)
        # If verification fails after all retries, raise an exception or handle it as needed
        raise Exception("Response verification failed after retries.")

    def safe_error_prompt(self, prompt, role_name, response, response_structure):

        # Save failed prompt as a .md file
        file_name = "__failed_prompt_" + role_name + ".md"
        output_filename = os.path.join(self.output_path, file_name)
        with open(output_filename, "w") as f:
            f.write(prompt)

        # Save failed response as a .md file
        file_name = "__failed_response_" + role_name + ".md"
        output_filename = os.path.join(self.output_path, file_name)
        with open(output_filename, "w") as f:
            f.write(response)

        # Save failed structure as a .md file
        file_name = "__failed__structure_" + role_name + ".md"
        output_filename = os.path.join(self.output_path, file_name)
        
        if isinstance(response_structure, dict):
            response_structure = json.dumps(response_structure)
            
        with open(output_filename, "w") as f:
            f.write(response_structure)

    

    def get_verified_response(self, role, prompts: List[str], response_format, response_structure) -> str:
        """
        Sends the prompt array to the GPT model for verification and receives verification feedback
        """
        responses = []
        
        i = 0
        role_name = fill_gaps_with_underscore(role)
        for prompt in prompts:
            i += 1

            print(f"{role} is working on: prompt {i} of {len(prompts)}")
            verified_response = self.get_verified_response_single(role, prompt, response_format, response_structure)

            # Save verification results as a .md file
            file_name = "prompt_" + role_name + ".md"
            file_name = file_name.replace("..", ".")
            output_filename = os.path.join(self.project_path, "output", file_name)
            with open(output_filename, "w") as f:
                f.write(prompt)

            # Save verification results as a .md file
            file_name = "response_" + role_name + ".md"
            file_name = file_name.replace("..", ".")
            output_filename = os.path.join(self.project_path, "output", file_name)
            with open(output_filename, "w") as f:
                f.write(verified_response)


            responses.append(verified_response)

        return "\n".join(responses)
    
    
    def submit_prompt(self, nlp_client, prompttext, role = "You are a award winning web developer", response_format = "json", response_structure = ""):
        # Use the get_verified_response method of your GPTModel class to send the prompttext for verification
        response = self.get_verified_response(role , [prompttext], response_format, response_structure)
        return response

    def init_folders(self):
        self.project_path = self.memory.read(["project", "files", "project_folder"])
        self.prompt_path = self.memory.read(["project", "files", "prompt_folder"])
        self.response_path = self.memory.read(["project", "files", "response_folder"])
        self.output_path = self.memory.read(["project", "files", "output_folder"])


    def get_response(self, role, prompt, response_format, response_structure = "", item_number = '', extract_json = True):
        
        self.init_folders()
        
        if item_number == '':
            file_suffix = ''
        else:
            file_suffix = "_" + item_number
        
        # Save prompt to a text file
        self.save_item(role, "Prompt", file_suffix, prompt)

        # Generate response
        response = self.submit_prompt(self.nlp_client, prompt, role, response_format, response_structure)
        
        if extract_json and response_format == "json": 
            response = extract_json_string_from_message(response)

        # Save response to a text file
        self.save_item(role, "Response", file_suffix, response)

        self.counter += 1

        return response
    
    def process_response(self, response):
        # Implement your response processing logic here
        processed_response = response  # Placeholder implementation, modify as needed
        return processed_response
    
    def to_json(self):
        # Serialize the NLPProcessor object to JSON.
        # This includes the serialization of the NLPClient.
        return json.dumps({
            "class": "NLPProcessor",
            "nlp_client": self.nlp_client.to_json()
        })

    @classmethod
    def from_json(cls, json_str):
        # Deserialize the JSON string back to an NLPProcessor object.
        data = json.loads(json_str)
        nlp_client = NLPClient.from_json(data["nlp_client"])
        return cls(nlp_client=nlp_client)


## nlp/nlp_client.py:

import time
import logging
import tiktoken
import os
import json
import openai
from openai import OpenAI

GPT_MODEL = "gpt-4-1106-preview"
GPT_MODEL = "gpt-3.5-turbo-1106"

RETRY_COUNT = 0
MAX_RETRIES = 50
WAIT_TIME = 5

class NLPClient:
    def __init__(self):
        pass

    def process(self, prompt, role):
        client = OpenAI()
        retry_count = RETRY_COUNT
        wait_time = WAIT_TIME


        while retry_count < MAX_RETRIES:
            try:
                #Make your OpenAI API request here
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": prompt},
                    ]
                )
                break

            except openai.APIError as e:
                logging.error(f"OpenAI API Error: {str(e)}")
                wait_time += 5
                print(f"OpenAI: Retrying in {wait_time} seconds, retry count: {retry_count}...")
                time.sleep(wait_time)
                retry_count += 1
                
            except openai.APIConnectionError as e:
                logging.error(f"OpenAI API Connection Error: {str(e)}")
                wait_time += 5
                print(f"OpenAI: Retrying in {wait_time} seconds, retry count: {retry_count}...")
                time.sleep(wait_time)
                retry_count += 1

            except openai.APITimeoutError as e:
                logging.error(f"OpenAI Timeout Error: {str(e)}")
                wait_time += 5
                print(f"OpenAI: Retrying in {wait_time} seconds, retry count: {retry_count}...")
                time.sleep(wait_time)
                retry_count += 1

            except openai.RateLimitError as e:
                logging.error(f"OpenAI Rate Limit Error (You have hit your assigned rate limit): {str(e)}")
                wait_time += 5
                print(f"OpenAI: Retrying in {wait_time} seconds, retry count: {retry_count}...")
                time.sleep(wait_time)
                retry_count += 1
                
            except openai.InternalServerError as e:
                logging.error(f"OpenAI Internal Server Error: {str(e)}")
                wait_time += 5
                print(f"OpenAI: Retrying in {wait_time} seconds, retry count: {retry_count}...")
                time.sleep(wait_time)
                retry_count += 1
                                
            except openai.AuthenticationError as e:
                logging.error(f"OpenAI Authentication Error: {str(e)}")
                break
            
            except openai.BadRequestError as e:
                logging.error(f"OpenAI Bad Request Error (Your request was malformed or missing some required parameters, such as a token or an input): {str(e)}")
                break
            
            except openai.ConflictError as e:
                logging.error(f"OpenAI Conflict Error (The resource was updated by another request): {str(e)}")
                break
            
            except openai.NotFoundError as e:
                logging.error(f"OpenAI Not Found Error (Requested resource does not exist.): {str(e)}")
                break
            
            except openai.PermissionDeniedError as e:
                logging.error(f"OpenAI Permission Denied Error (You don't have access to the requested resource.): {str(e)}")
                break
                        
            except openai.UnprocessableEntityError as e:
                logging.error(f"OpenAI Unprocessable Entity Error (Unable to process the request despite the format being correct): {str(e)}")
                break
            
            except Exception as e:
                logging.error(f"Unexpected Error: {str(e)}")
                wait_time += 5
                print(f"OpenAI: Retrying in {wait_time} seconds, retry count: {retry_count}...")
                time.sleep(wait_time)
                retry_count += 10
                
            retry_count += 1

        response = response.choices[0].message.content
        
        print ("prompt processed by NLPClient")
        
        return response

    def process_as_json(self, prompt, role):
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ]
        )

        return response.choices[0].message.content

    def to_json(self):
        # Serialize the NLPClient object to JSON.
        # Note: since this class does not contain any dynamic data, 
        # we return a basic representation.
        return json.dumps({"class": "NLPClient"})

    @classmethod
    def from_json(cls, json_str):
        # Deserialize the JSON string back to an NLPClient object.
        # Since there's no dynamic data, we simply return a new instance.
        return cls()

## api/api_key_loader.py:

import os
import logging
from dotenv import load_dotenv

class OpenAIKeyLoader:
    def __init__(self, dotenv_path=None):

        # Check if the .env file exists before trying to load it
        if dotenv_path is None or not os.path.exists(dotenv_path):
            raise FileNotFoundError(f"The .env file was not found at {dotenv_path}")

        # Load environment variables from the .env file
        load_dotenv(dotenv_path)

        # Get the OpenAI API Key
        self.api_key = os.environ.get("OPENAI_API_KEY")

        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("API key not found in .env file.")
            self.api_key = api_key
            logging.info("API Key loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading API key: {e}. Please check the .env file.")
            raise

    def get_api_key(self):
        return self.api_key


## LICENSE:

MIT License

Copyright (c) 2024 mxn2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## maldinio_ai.egg-info/PKG-INFO:

Metadata-Version: 2.1
Name: maldinio-ai
Version: 0.1.1
Summary: A utility package for AI prompt management and prompt processing.
Author: Mehdi Nabhani
Author-email: mehdi@nabhani.de
Keywords: AI,NLP,LLM,Prompt Management
License-File: LICENSE


## maldinio_ai.egg-info/SOURCES.txt:

LICENSE
README.md
setup.py
api/__init__.py
api/api_key_loader.py
maldinio_ai.egg-info/PKG-INFO
maldinio_ai.egg-info/SOURCES.txt
maldinio_ai.egg-info/dependency_links.txt
maldinio_ai.egg-info/requires.txt
maldinio_ai.egg-info/top_level.txt
memory_management/__init__.py
memory_management/memory_manager.py
nlp/__init__.py
nlp/nlp_client.py
nlp/nlp_processor.py
prompt/__init__.py
prompt/prompt_context.py
prompt/prompt_generator.py
prompt/response_processor.py
tools/__init__.py
tools/create_project_folder.py
tools/load_project.py
utils/__init__.py
utils/helpers.py
utils/json_utils.py
utils/verification_utils.py

## maldinio_ai.egg-info/requires.txt:

tiktoken>=0.4.0
openai>=1.3.7


## maldinio_ai.egg-info/top_level.txt:

api
memory_management
nlp
prompt
tools
utils


## maldinio_ai.egg-info/dependency_links.txt:




## File and Folder Structure:

{
    "": [
        "LICENSE",
        "README.md",
        "setup.py"
    ],
    "tools": [
        "create_project_folder.py",
        "load_project.py"
    ],
    "utils": [
        "verification_utils.py",
        "json_utils.py",
        "helpers.py"
    ],
    "prompt": [
        "prompt_context.py",
        "prompt_generator.py",
        "response_processor.py"
    ],
    "memory_management": [
        "memory_manager.py"
    ],
    "nlp": [
        "nlp_processor.py",
        "nlp_client.py"
    ],
    "api": [
        "api_key_loader.py"
    ],
    "maldinio_ai.egg-info": [
        "PKG-INFO",
        "SOURCES.txt",
        "requires.txt",
        "top_level.txt",
        "dependency_links.txt"
    ]
}