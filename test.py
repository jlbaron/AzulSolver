'''
Test trained agents here
This file will include some pygame code for visualization
TODO: make a version where a human can interact with the AI
'''
import yaml

# Assuming your YAML content is stored in a file called 'config.yaml'
with open('configs\config.yaml', 'r') as file:
    data = yaml.safe_load(file)

# The data variable now contains the dictionaries as you defined them.
# Accessing the dictionaries
hyperparameters = data['hyperparameters']
game_info = data['game_info']

# You can now use hyperparameters and game_info just like you did with your original code
print(hyperparameters)
print(game_info)