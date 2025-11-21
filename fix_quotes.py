"""Fix escaped triple quotes in gradio_app.py"""
import re

file_path = r'src\ai_workdesk\ui\gradio_app.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace escaped triple quotes with normal triple quotes
content = content.replace(r'\"\"\"', '"""')

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed escaped triple quotes!")
