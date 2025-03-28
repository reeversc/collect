import os
import time
from anthropic import Anthropic

class PassageAnalysisAgent:
    def __init__(self, model="claude-3-7-sonnet-20250219", api_key=None, max_iterations=4):
        """Initialize a self-reflective agent that can deeply analyze a passage from Wikipedia."""
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.conversation_history = []
        
        # Define the think tool
        self.think_tool = {
            "name": "think",
            "description": "Use this tool to deeply analyze the provided passage, examining its claims, implications, and connections to broader knowledge. The thinking process should build on previous reflections.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "A detailed analysis or reflection on the passage."
                    }
                },
                "required": ["thought"]
            }
        }
        
        # Initialize variables
        self.passage = None
        self.topic = None
    
    def set_passage(self, passage, topic=None):
        """Set the passage to be analyzed and optionally a specific topic focus."""
        self.passage = passage
        self.topic = topic if topic else "Deep analysis of the provided passage"
        
        # System prompt for passage analysis
        self.system_prompt = f"""
        You are an analytical agent who deeply examines scientific passages. You've been provided 
        with the following passage from a Wikipedia article:
        
        "{self.passage}"
        
        Use the 'think' tool to perform careful, rigorous analysis of this passage. Your analysis should:
        
        1. Break down key claims and assertions in the passage
        2. Examine the scientific concepts and mechanisms described
        3. Identify underlying assumptions or implications
        4. Connect the information to broader scientific understanding
        5. Consider limitations or gaps in what's presented
        
        Build upon your previous thinking in each iteration, progressively deepening your understanding.
        Focus on analytical insight rather than mere summary or paraphrase.
        """
        
        # Reset conversation history
        self.conversation_history = []
    
    def start_analysis(self):
        """Begin the analysis process on the current passage."""
        if not self.passage:
            raise ValueError("You must set a passage before starting analysis. Use set_passage() method.")
            
        print(f"🔍 Beginning analysis of passage about {self.topic}\n")
        print("=" * 80)
        
        # Initial analysis prompt
        initial_prompt = "Begin analyzing the passage, focusing first on identifying and examining the key scientific claims and concepts it presents."
        
        # Add initial prompt to conversation history
        self.conversation_history = [
            {"role": "user", "content": initial_prompt}
        ]
        
        # Run self-reflection for specified number of iterations
        for i in range(self.max_iterations):
            print(f"\n📝 Analysis Iteration {i+1}:\n")
            
            # Get response from Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=[self.think_tool]
            )
            
            # Process the response
            continue_analysis = self.process_response(response, i)
            if not continue_analysis:
                break
            
            # Pause between iterations to simulate thinking time
            time.sleep(1.5)
            
        print("\n" + "=" * 80)
        print("🧠 Passage analysis complete.")
        
        # Return the final conversation history for potential further use
        return self.conversation_history
    
    def process_response(self, response, iteration):
        """Process Claude's response, handling any tool usage."""
        # Check if Claude is using the think tool
        if response.stop_reason == "tool_use":
            # Find the tool use block
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "think":
                    tool_use_block = block
                    break
            
            if tool_use_block:
                # Extract the thinking content
                thought = tool_use_block.input.get("thought", "")
                print("💭 Analysis:")
                print("-" * 50)
                print(thought)
                print("-" * 50 + "\n")
                
                # Add the assistant's response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response.content})
                
                # Create tool result message
                self.conversation_history.append({
                    "role": "user", 
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": thought
                        }
                    ]
                })
                
                # Generate the next prompt based on the current iteration
                next_prompt = self.generate_next_prompt(iteration)
                
                self.conversation_history.append({
                    "role": "user",
                    "content": next_prompt
                })
                
                return True
        else:
            # If no tool used, just display the response
            for content_block in response.content:
                if content_block.type == "text":
                    print("💬 Response:")
                    print("-" * 50)
                    print(content_block.text)
                    print("-" * 50 + "\n")
            
            # Add the response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response.content})
            
            # Generate the next prompt
            next_prompt = self.generate_next_prompt(iteration)
            
            self.conversation_history.append({
                "role": "user",
                "content": next_prompt
            })
            
            return True
        
        return False
    
    def generate_next_prompt(self, iteration):
        """Generate the next prompt based on the iteration number."""
        # Progression of analysis prompts specific to passage analysis
        prompts = [
            "Now that you've identified the key concepts, examine the biological processes and mechanisms described in the passage. How do they work, and what are their implications?",
            
            "Based on your analysis of the mechanisms, consider what's missing or implied but not explicitly stated in the passage. What scientific context or background knowledge helps complete the picture?",
            
            "Synthesize your previous analyses to evaluate the potential significance of this information. Consider both scientific and practical implications, limitations, and future research directions."
        ]
        
        # Use a standard prompt if we have one for this iteration
        if iteration < len(prompts):
            return prompts[iteration]
        
        # Otherwise, return a generic continuation prompt
        return "Continue your analysis, exploring aspects of the passage that warrant deeper investigation."
    
    def save_analysis(self, filename=None):
        """Save the complete analysis to a text file."""
        if not self.passage or not self.conversation_history:
            raise ValueError("No analysis to save. Run start_analysis() first.")
            
        if filename is None:
            # Create a filename based on the topic
            safe_topic = "".join(x for x in self.topic if x.isalnum() or x in [' ', '_']).replace(' ', '_')
            filename = f"{safe_topic}_analysis.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Analysis of passage about: {self.topic}\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"PASSAGE:\n{self.passage}\n\n")
            f.write("=" * 80 + "\n\n")
            
            # Extract and write thinking content
            iteration = 1
            for i in range(len(self.conversation_history)):
                message = self.conversation_history[i]
                
                if message["role"] == "user" and isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if content_item.get("type") == "tool_result":
                            f.write(f"ANALYSIS ITERATION {iteration}:\n")
                            f.write("-" * 50 + "\n")
                            f.write(content_item.get("content", "") + "\n\n")
                            iteration += 1
            
            f.write("=" * 80 + "\n")
            f.write(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Analysis saved to {filename}")
        return filename


# Example usage with the Urolithin A passage from Wikipedia
if __name__ == "__main__":
    # Create the agent
    agent = PassageAnalysisAgent()
    
    # The passage from Wikipedia about Urolithin A
    urolithin_passage = """
    Urolithin A is a metabolite compound resulting from the transformation of ellagitannins by the gut bacteria. 
    It belongs to the class of organic compounds known as benzo-coumarins or dibenzo-α-pyrones. 
    Its precursors – ellagic acids and ellagitannins – are ubiquitous in nature, including edible plants, 
    such as pomegranates, strawberries, raspberries, walnuts, and others. Urolithin A is not known to be 
    found in any food source. Its bioavailability mostly depends on individual microbiota composition, 
    as only some bacteria are able to convert ellagitannins into urolithins.
    
    When synthesized and absorbed in the intestines, urolithin A enters the systemic circulation where 
    it becomes available to tissues throughout the body where it is further subjected to additional 
    chemical transformations (including glucuronidation, methylation, sulfation, or a combination of them) 
    within the enterocytes and hepatocytes. Urolithin A and its derivatives - urolithin A glucuronide 
    and urolithin A sulfate being most abundant - release into the circulation, before being excreted 
    in the urine. In vivo studies did not determine any toxicity or specific adverse effects following 
    dietary intake of urolithin A. Safety studies in elderly humans indicated urolithin A was well 
    tolerated. In 2018, the US Food and Drug Administration listed urolithin A as a safe ingredient 
    for food products having content in the range of 250 mg to one gram per serving.
    """
    
    # Set the passage and topic
    agent.set_passage(urolithin_passage, "Urolithin A metabolism and safety")
    
    # Start the analysis
    agent.start_analysis()
    
    # Optionally save the analysis to a file
    # agent.save_analysis("urolithin_a_analysis.txt")
