from crewai import Agent, Crew, Process, Task
from crewai_tools import DataAnalystTool, LifeCoachTool, ResearcherTool, InterfaceTool, DeciderTool
from crewai_tools.memory import SharedMemory

# Shared memory across all agents
shared_memory = SharedMemory()

# Define the Data Analyst Task
class DataAnalystTask(Task):
    def __init__(self):
        super().__init__(name="Analyze Physiometric Data")
        self.tool = DataAnalystTool()

    def execute(self, data_stream):
        try:
            insights = []
            if "heart_rate" in data_stream:
                insights.append(self.tool.analyze_heart_rate(data_stream["heart_rate"]))
            if "sleep_data" in data_stream:
                insights.append(self.tool.analyze_sleep(data_stream["sleep_data"]))
            return insights
        except Exception as e:
            return f"Data analysis failed: {e}"

# Define the Life Coach Task
class LifeCoachTask(Task):
    def __init__(self):
        super().__init__(name="Offer Mental Health Advice")
        self.tool = LifeCoachTool()
        self.memory = shared_memory

    def execute(self, user_input):
        try:
            previous_advice = self.memory.get("last_advice")
            response = self.tool.advise(user_input, previous_advice)
            self.memory.set("last_advice", response)
            return response
        except Exception as e:
            return f"Life coaching failed: {e}"

# Define the Researcher Task
class ResearcherTask(Task):
    def __init__(self):
        super().__init__(name="Perform Health-Related Research")
        self.tool = ResearcherTool()

    def execute(self, query):
        try:
            research_results = self.tool.search(query)
            return research_results
        except Exception as e:
            return f"Research failed: {e}"

# Define the Interface Task to integrate all other agents' results
class InterfaceTask(Task):
    def __init__(self, data_analyst_task, life_coach_task, researcher_task):
        super().__init__(name="Integrate Health Insights and Recommendations")
        self.data_analyst_task = data_analyst_task
        self.life_coach_task = life_coach_task
        self.researcher_task = researcher_task

    def execute(self, data, user_input, query):
        try:
            data_insights = self.data_analyst_task.execute(data)
            coach_response = self.life_coach_task.execute(user_input)
            research_info = self.researcher_task.execute(query)
            return (
                f"**Health Insights**: {', '.join(data_insights)}\n"
                f"**Life Coach Advice**: {coach_response}\n"
                f"**Research Info**: {research_info}"
            )
        except Exception as e:
            return f"Integration failed: {e}"

# Define the Decider Task that routes tasks based on user input
class DeciderTask(Task):
    def __init__(self, interface_task):
        super().__init__(name="Delegate Tasks Based on User Input")
        self.interface_task = interface_task

    def execute(self, user_input, data, query):
        try:
            if "stress" in user_input or "sleep" in query:
                return self.interface_task.execute(data, user_input, query)
            else:
                return "No relevant agent to handle this request."
        except Exception as e:
            return f"Task delegation failed: {e}"

# Define the Data Analyst Agent
class DataAnalystAgent(Agent):
    def __init__(self):
        super().__init__(
            name="DataAnalystAgent",
            tasks=[DataAnalystTask()],
            description="Analyzes physiometric data from wearables and generates insights.",
            memory=shared_memory
        )

# Define the Life Coach Agent
class LifeCoachAgent(Agent):
    def __init__(self):
        super().__init__(
            name="LifeCoachAgent",
            tasks=[LifeCoachTask()],
            description="Provides emotional and mental wellness advice.",
            memory=shared_memory
        )

# Define the Researcher Agent
class ResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ResearcherAgent",
            tasks=[ResearcherTask()],
            description="Performs research and fetches the latest relevant information.",
            memory=shared_memory
        )

# Define the Interface Agent
class InterfaceAgent(Agent):
    def __init__(self, data_analyst_task, life_coach_task, researcher_task):
        super().__init__(
            name="InterfaceAgent",
            tasks=[InterfaceTask(data_analyst_task, life_coach_task, researcher_task)],
            description="Integrates the results from other agents and communicates with the user.",
            memory=shared_memory
        )

# Define the Decider Agent
class DeciderAgent(Agent):
    def __init__(self, interface_task):
        super().__init__(
            name="DeciderAgent",
            tasks=[DeciderTask(interface_task)],
            description="Routes tasks to the appropriate agents and coordinates decision-making.",
            memory=shared_memory
        )

# Main function to create crew and run a sample process
def main():
    # Initialize the tasks
    data_analyst_task = DataAnalystTask()
    life_coach_task = LifeCoachTask()
    researcher_task = ResearcherTask()
    interface_task = InterfaceTask(data_analyst_task, life_coach_task, researcher_task)

    # Initialize the agents
    data_analyst = DataAnalystAgent()
    life_coach = LifeCoachAgent()
    researcher = ResearcherAgent()
    interface = InterfaceAgent(data_analyst_task, life_coach_task, researcher_task)
    decider = DeciderAgent(interface_task)

    # Create a crew with agents and tasks
    wellness_crew = Crew(
        agents=[data_analyst, life_coach, researcher, interface, decider],
        tasks=[data_analyst_task, life_coach_task, researcher_task, interface_task, decider.tasks[0]],
        process=Process.sequential,
        name="WellAware"
    )

    # Example inputs for the system
    data_stream = {"heart_rate": [60, 75, 85], "sleep_data": [7, 6.5, 8]}
    user_question = "I've been feeling stressed lately. What should I do?"
    research_query = "Latest research on the impact of sleep on stress."

    # Execute the decider task
    result = decider.tasks[0].execute(user_question, data_stream, research_query)
    print(result)

if __name__ == "__main__":
    main()
