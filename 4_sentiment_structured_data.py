from anthropic import Anthropic
import json
import os


    
tools = [
    {
        "name": "print_sentiment_scores",
        "description": "Prints the sentiment scores of a given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "positive_score": {"type": "number", "description": "The positive sentiment score, ranging from 0.0 to 1.0."},
                "negative_score": {"type": "number", "description": "The negative sentiment score, ranging from 0.0 to 1.0."},
                "neutral_score": {"type": "number", "description": "The neutral sentiment score, ranging from 0.0 to 1.0."}
            },
            "required": ["positive_score", "negative_score", "neutral_score"]
        }
    }
]

api_key = os.getenv('ANTHROPIC_API_KEY')    
client = Anthropic(api_key=api_key)

def analyze_sentiment(content, print_result=True):
    query = f"""
    <text>
    {content}
    </text>

    Only use the print_sentiment_scores tool.
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        tools=tools,
        # we can force a tool to be used by adding the tool_choice parameter
        tool_choice={"type": "tool", "name": "print_sentiment_scores"}, # can also be "auto" or "any"
        messages=[{"role": "user", "content": query}]
    )

    json_sentiment = None
    for content in response.content:
        if content.type == "tool_use" and content.name == "print_sentiment_scores":
            json_sentiment = content.input
            break

    if json_sentiment:
        if print_result:
            print("Sentiment Analysis (JSON):")
            print(json.dumps(json_sentiment, indent=2))
        return json_sentiment
    else:
        if print_result:
            print("No sentiment analysis found in the response.")
        return None

# Example usage
analyze_sentiment("OMG I absolutely love taking bubble baths soooo much!!!!")
analyze_sentiment("Honestly I have no opinion on taking baths")

# Using the function to get the result without printing
result = analyze_sentiment("This movie was terrible!", print_result=False)
if result:
    positive_score = result['positive_score']
    negative_score = result['negative_score']
    neutral_score = result['neutral_score']
    print(f"Positive: {positive_score}, Negative: {negative_score}, Neutral: {neutral_score}")
