import json
from anthropic import Anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

translate_tool = {
    "name": "translate",
    "description": "Translates a phrase into multiple languages",
    "input_schema": {
        "type": "object",
        "properties": {
            "english": {"type": "string"},
            "spanish": {"type": "string"},
            "french": {"type": "string"},
            "japanese": {"type": "string"},
            "arabic": {"type": "string"}
        },
        "required": ["english", "spanish", "french", "japanese", "arabic"]
    }
}

def translate(phrase):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": f"Translate '{phrase}' to Spanish, French, Japanese, and Arabic."}],
        max_tokens=500,
        tools=[translate_tool],
        tool_choice={"type": "tool", "name": "translate"} # can be {"type": "auto"} or {"type": "any"}
        # auto allows Claude to decide whether to call any provided tools or not.
        # any tells Claude that it must use one of the provided tools, but doesn't force a particular tool.
        # tool allows us to force Claude to always use a particular tool.
    )
    
    for content in response.content:
        if content.type == "tool_use" and content.name == "translate":
            return content.input

    return None

# Example usage
translations = translate("how much does this cost")
print(json.dumps(translations, ensure_ascii=False, indent=2))