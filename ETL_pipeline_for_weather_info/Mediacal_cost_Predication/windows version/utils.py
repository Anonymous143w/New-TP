import os

# Create directories
directories = [
    'data/raw',
    'data/processed',
    'models',
    'logs'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)