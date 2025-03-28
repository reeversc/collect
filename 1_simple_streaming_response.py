from anthropic import Anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")    
client = Anthropic(api_key=api_key)

# A relatively simple math problem
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content":"Multiply 1984135 by 9343116. Only respond with the result"}],
    max_tokens=4000,
    stream=True
)

assistant_response = ""
# Print the streaming responses
for chunk in response:
    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
        assistant_response += chunk.delta.text
        print(chunk.delta.text, end='', flush=True)

