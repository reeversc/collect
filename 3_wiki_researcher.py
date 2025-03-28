from anthropic import Anthropic
import os
import wikipedia

api_key = os.getenv("ANTHROPIC_API_KEY")    
client = Anthropic(api_key=api_key)


def generate_wikipedia_reading_list(research_topic, article_titles):
    wikipedia_articles = []
    for t in article_titles:
        print(f"Searching for {t}")
        results = wikipedia.search(t)
        try:
            page = wikipedia.page(results[0])
            title = page.title
            url = page.url
            wikipedia_articles.append({"title": title, "url": url})
        except:
            continue
    add_to_research_reading_file(wikipedia_articles, research_topic)

def add_to_research_reading_file(articles, topic):
    os.makedirs("output", exist_ok=True)  # Create the output directory if it doesn't exist
    with open("output/research_reading.md", "a", encoding="utf-8") as file:
        file.write(f"## {topic} \n")
        for article in articles:
            title = article["title"]
            url = article["url"]
            file.write(f"* [{title}]({url}) \n")
        file.write(f"\n\n")

# IMPORTANT: SEE HOW WE ARE MENTIONING THAT USER SPECIFIES NUMBER OF ARTICLES TO GENERATE
wikipedia_tool = {
    "name": "generate_wikipedia_reading_list",
    "description": "Generates a list of Wikipedia articles based on a research topic and potential article titles. How many articles to generate is specified by the user.",
    "input_schema": {
        "type": "object",
        "properties": {
            "research_topic": {"type": "string", "description": "The research topic."},
            "article_titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of n many potential Wikipedia article titles. n is specified by the user."
            }
        },
        "required": ["research_topic", "article_titles"],
    },
}

def get_research_help(topic, num_articles=3):
    prompt = f"Generate a list of {num_articles} Wikipedia article titles related to the topic '{topic}'."
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system="You have access to tools, but only use them when necessary. If a tool is not required, respond as normal",
        messages=messages,
        max_tokens=500,
        tools=[wikipedia_tool],
    )

    if response.stop_reason == "tool_use":
        tool_use = response.content[-1]
        tool_name = tool_use.name
        tool_input = tool_use.input

        if tool_name == "generate_wikipedia_reading_list":
            print("Claude wants to use the Wikipedia tool")
            research_topic = tool_input["research_topic"]
            article_titles = tool_input["article_titles"]

            generate_wikipedia_reading_list(research_topic, article_titles)

    elif response.stop_reason == "end_turn":
        print("Claude didn't want to use a tool")
        print("Claude responded with:")
        print(response.content[0].text)

# Example function calls
get_research_help("Pirates Across The World", 3)
get_research_help("History of Hawaii", 2)
get_research_help("are animals conscious?", 4)