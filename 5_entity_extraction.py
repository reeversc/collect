import os
from anthropic import Anthropic
import json

api_key = os.getenv('ANTHROPIC_API_KEY')
client = Anthropic(api_key=api_key)

tools = [
    {
        "name": "print_entities",
        "description": "Prints extracted named entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The extracted entity name."},
                            "type": {"type": "string", "description": "The entity type (e.g., PERSON, ORGANIZATION, LOCATION)."},
                            "context": {"type": "string", "description": "The context in which the entity appears in the text."}
                        },
                        "required": ["name", "type", "context"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
]

text = "John works at Google in New York. He met with Sarah, the CEO of Acme Inc., last week in San Francisco."

query = f"""
<document>
{text}
</document>

Use the print_entities tool.
"""

response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

json_entities = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_entities":
        json_entities = content.input
        break

if json_entities:
    print("Extracted Entities (JSON):")
    print(json.dumps(json_entities, indent=2))
else:
    print("No entities found in the response.")