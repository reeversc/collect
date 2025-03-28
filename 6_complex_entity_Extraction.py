import wikipedia
import json
from anthropic import Anthropic
import os

# Define the tool for article classification
tools = [
    {
        "name": "print_article_classification",
        "description": "Prints the classification results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The overall subject of the article",
                },
                "summary": {
                    "type": "string",
                    "description": "A paragaph summary of the article"
                },
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "List of keywords and topics in the article"
                    }
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The category name."},
                            "score": {"type": "number", "description": "The classification score for the category, ranging from 0.0 to 1.0."}
                        },
                        "required": ["name", "score"]
                    }
                }
            },
            "required": ["subject","summary", "keywords", "categories"]
        }
    }
]

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Function to generate JSON for a given article subject
def generate_json_for_article(subject):
    # Fetch Wikipedia page content
    page = wikipedia.page(subject, auto_suggest=True)
    
    # Prepare query for Claude
    query = f"""
    <document>
    {page.content}
    </document>

    Use the print_article_classification tool. Example categories: Politics, Sports, Technology, Entertainment, Business.
    """

    # Send request to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        tools=tools,
        messages=[{"role": "user", "content": query}]
    )

    # Extract and print JSON classification
    for content in response.content:
        if content.type == "tool_use" and content.name == "print_article_classification":
            print(json.dumps(content.input, indent=2))
            return

    print("No text classification found in the response.")

# Example usage
generate_json_for_article("Jeff Goldblum")
generate_json_for_article("Octopus")