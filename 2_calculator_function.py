from anthropic import Anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")    
client = Anthropic(api_key=api_key)

def calculator(operation, operand1, operand2):
    if operation == "add":
        return operand1 + operand2
    elif operation == "subtract":
        return operand1 - operand2
    elif operation == "multiply":
        return operand1 * operand2
    elif operation == "divide":
        if operand2 == 0:
            raise ValueError("Cannot divide by zero.")
        return operand1 / operand2
    else:
        raise ValueError(f"Unsupported operation: {operation}")


calculator_tool = {
    "name": "calculator",
    "description": "A simple calculator that performs basic arithmetic operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform.",
            },
            "operand1": {"type": "number", "description": "The first operand."},
            "operand2": {"type": "number", "description": "The second operand."},
        },
        "required": ["operation", "operand1", "operand2"],
    },
}


def prompt_claude(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system="You have access to tools, but only use them when necessary. If a tool is not required, respond as normal",
        messages=messages,
        max_tokens=500,
        tools=[calculator_tool],
    )

    if response.stop_reason == "tool_use":
        tool_use = response.content[-1]
        tool_name = tool_use.name
        tool_input = tool_use.input

        if tool_name == "calculator":
            print("Claude wants to use the calculator tool")
            operation = tool_input["operation"]
            operand1 = tool_input["operand1"]
            operand2 = tool_input["operand2"]

            try:
                result = calculator(operation, operand1, operand2)
                print("Calculation result is:", result)
            except ValueError as e:
                print(f"Error: {str(e)}")

    elif response.stop_reason == "end_turn":
        print("Claude didn't want to use a tool")
        print("Claude responded with:")
        print(response.content[0].text)

prompt_claude("I had 23 chickens but 2 flew away.  How many are left?")
prompt_claude("What is 201 times 2")
prompt_claude("Write me a haiku about the ocean")