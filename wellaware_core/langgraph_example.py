# Import necessary modules
from langchain import LLMMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langgraph import GraphNode, Graph

# Create common LLM model for tasks using chat-gpt-4o-mini
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

# Define the Data Analyst Task as a Graph Node
class DataAnalystNode(GraphNode):
    def __init__(self):
        super().__init__(name="DataAnalystNode")

    def execute(self, data_stream):
        template = "You are a Data Analyst. Analyze the following data and provide insights: {data}"
        prompt = PromptTemplate(input_variables=["data"], template=template)
        chain = LLMChain(prompt=prompt, llm=llm)
        return chain.run({"data": data_stream})

# Define the Life Coach Task as a Graph Node
class LifeCoachNode(GraphNode):
    def __init__(self):
        super().__init__(name="LifeCoachNode")

    def execute(self, user_input):
        template = "You are a Life Coach. The user said: {user_input}. Provide mental wellness advice."
        prompt = PromptTemplate(input_variables=["user_input"], template=template)
        chain = LLMChain(prompt=prompt, llm=llm)
        return chain.run({"user_input": user_input})

# Define the Researcher Task as a Graph Node
class ResearcherNode(GraphNode):
    def __init__(self):
        super().__init__(name="ResearcherNode")

    def execute(self, query):
        template = "You are a Researcher. Search for relevant information based on: {query}."
        prompt = PromptTemplate(input_variables=["query"], template=template)
        chain = LLMChain(prompt=prompt, llm=llm)
        return chain.run({"query": query})

# Define the Interface Task as a Graph Node
class InterfaceNode(GraphNode):
    def __init__(self, data_analyst_node, life_coach_node, researcher_node):
        super().__init__(name="InterfaceNode")
        self.data_analyst_node = data_analyst_node
        self.life_coach_node = life_coach_node
        self.researcher_node = researcher_node

    def execute(self, data, user_input, query):
        # Get results from the other nodes
        data_insights = self.data_analyst_node.execute(data)
        life_coach_advice = self.life_coach_node.execute(user_input)
        research_info = self.researcher_node.execute(query)

        return (f"**Data Insights**: {data_insights}\n"
                f"**Life Coach Advice**: {life_coach_advice}\n"
                f"**Research Information**: {research_info}")

# Define the Decider Task as a Graph Node
class DeciderNode(GraphNode):
    def __init__(self, interface_node):
        super().__init__(name="DeciderNode")
        self.interface_node = interface_node

    def execute(self, user_input, data_stream, query):
        # Route to the interface node for integrating responses
        return self.interface_node.execute(data_stream, user_input, query)

# Main function to create the graph and execute the nodes
def main():
    # Initialize the nodes
    data_analyst_node = DataAnalystNode()
    life_coach_node = LifeCoachNode()
    researcher_node = ResearcherNode()
    interface_node = InterfaceNode(data_analyst_node, life_coach_node, researcher_node)
    decider_node = DeciderNode(interface_node)

    # Create the graph and define the relationships between nodes
    graph = Graph()
    graph.add_node(decider_node)
    graph.add_node(data_analyst_node)
    graph.add_node(life_coach_node)
    graph.add_node(researcher_node)
    graph.add_node(interface_node)

    # Example inputs for the system
    data_stream = "Heart rate: 60, 75, 85. Sleep hours: 7, 6.5, 8."
    user_question = "I've been feeling stressed lately. What should I do?"
    research_query = "Latest research on the impact of sleep on stress."

    # Execute the decider node which routes to other nodes
    result = decider_node.execute(user_question, data_stream, research_query)
    print(result)

if __name__ == "__main__":
    main()
