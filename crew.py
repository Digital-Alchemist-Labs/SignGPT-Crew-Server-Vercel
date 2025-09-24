from crewai import Crew, Agent, Task
from crewai.project import CrewBase, crew, agent, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from dotenv import load_dotenv

load_dotenv()


@CrewBase
class SginGPTCrew:
  """Crew for SignGPT"""

  agents: List[BaseAgent]
  tasks: List[Task]

  # def __init__(self, asl_dataset):
  #   self.ASL_dataset = asl_dataset

  @agent
  def sentence_finisher_agent(self):
    """sentence finisher"""
    return Agent(
        config=self.agents_config['sentence_finisher_agent'],
        verbose=True,

    )

  @agent
  def chat_model_agent(self):
    """chat model"""
    return Agent(
        config=self.agents_config['chat_model_agent'],
        verbose=True,

    )

  @agent
  def sentence_splitter_agent(self):
    """sentence_splitter_agent"""
    return Agent(
        config=self.agents_config['sentence_splitter_agent'],
        verbose=True,

    )

  @task
  def finish_sentence_task(self):
    """finish sentence task"""
    return Task(
        config=self.tasks_config['finish_sentence_task'],
        verbose=True,
        # ASL_dataset=self.ASL_dataset,
    )

  @task
  def chat_task(self):
    """chat task"""
    return Task(
        config=self.tasks_config['chat_task'],
        verbose=True,
        # ASL_dataset=self.ASL_dataset,
    )

  @task
  def sentence_split_task(self):
    """sentence split task"""
    return Task(
        config=self.tasks_config['sentence_split_task'],
        verbose=True,
        # ASL_dataset=self.ASL_dataset,
    )

  @task
  def rearrange_word_task(self):
    """rearrange word task"""
    return Task(
        config=self.tasks_config['rearrange_word_task'],
        verbose=True,
        # ASL_dataset=self.ASL_dataset,
    )

  @crew
  def sgin_gpt_crew(self) -> Crew:
    """SignGPT Crew"""
    # Build agents
    finisher = self.sentence_finisher_agent()
    chatter = self.chat_model_agent()
    splitter = self.sentence_splitter_agent()

    # Build tasks
    t1 = self.finish_sentence_task()
    t2 = self.chat_task()
    t3 = self.sentence_split_task()
    t4 = self.rearrange_word_task()

    # Wire explicit context chaining: t1 -> t2 -> t3 -> t4
    t2.context = [t1]
    t3.context = [t2]
    t4.context = [t3]

    return Crew(
        agents=[finisher, chatter, splitter],
        # tasks=[t1, t2, t3, t4],
        tasks=[t1, t2, t3],
        verbose=True,
    )
