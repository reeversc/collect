import wikipedia
from anthropic import Anthropic
import re
import os


api_key = os.getenv('ANTHROPIC_API_KEY')
client = Anthropic(api_key=api_key)

def get_article(search_term):
    results = wikipedia.search(search_term)
    first_result = results[0]
    page = wikipedia.page(first_result, auto_suggest=False)
    return page.content

article_search_tool = {
    "name": "get_article",
    "description": "A tool to retrieve an up to date Wikipedia article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "search_term": {
                "type": "string",
                "description": "The search term to find a wikipedia article by title"
            },
        },
        "required": ["search_term"]
    }
}

def answer_question(question, messages):
    system_prompt = """
    You will be asked a question by the user. 
    If answering the question requires data you were not trained on, you can use the get_article tool to get the contents of a recent wikipedia article about the topic. 
    If you can answer the question without needing to get more information, please do so. 
    Only call the tool when needed. 
    You may use the tool multiple times if necessary to gather all required information.
    """
    prompt = f"""
    Answer the following question <question>{question}</question>
    When you can answer the question, keep your answer as short as possible and enclose it in <answer> tags
    """
    messages.append({"role": "user", "content": prompt})

    while True:
        if len(messages) >= 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
            messages.insert(-1, {"role": "assistant", "content": "analyzing to answer user request"})
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            system=system_prompt, 
            messages=messages,
            max_tokens=1000,
            tools=[article_search_tool]
        )
        
        if response.stop_reason != "tool_use":
            answer = re.search(r'<answer>(.*?)</answer>', response.content[0].text, re.DOTALL)
            return answer.group(1) if answer else response.content[0].text

        elif response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            for tool_use in response.content:
                if tool_use.type == "tool_use":
                    if tool_use.name == "get_article":
                        search_term = tool_use.input["search_term"]
                        print(f"Claude wants to get an article for {search_term}")
                        wiki_result = get_article(search_term)
                        tool_response = {
                            "role": "user",
                            "content": [
                                {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": wiki_result
                                }
                            ]
                        }
                        messages.append(tool_response)

def chatbot():
    messages = []
    print("Welcome to the AI Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        answer = answer_question(user_input, messages)
        print(f"AI: {answer}")

if __name__ == "__main__":
    chatbot()