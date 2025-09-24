from pathlib import Path
from crewai import Crew, Agent, Task
from crewai.project import CrewBase, crew, agent, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
import yaml

load_dotenv()

API_DIR = Path(__file__).resolve().parent
CONFIG_DIR = API_DIR / "config"
AGENTS_YAML = CONFIG_DIR / "agents.yaml"
TASKS_YAML  = CONFIG_DIR / "tasks.yaml"

@CrewBase
class SginGPTCrew:
  """Crew for SignGPT"""
  agents: List[BaseAgent]
  tasks: List[Task]

  # ✅ 추가: 서버리스에서도 확실히 YAML을 로드해 configs에 주입
  def __init__(self):
    with AGENTS_YAML.open("r", encoding="utf-8") as f:
      self.agents_config = yaml.safe_load(f) or {}
    with TASKS_YAML.open("r", encoding="utf-8") as f:
      self.tasks_config = yaml.safe_load(f) or {}

  @agent
  def sentence_finisher_agent(self):
    return Agent(config=self.agents_config['sentence_finisher_agent'], verbose=True)

  @agent
  def chat_model_agent(self):
    return Agent(config=self.agents_config['chat_model_agent'], verbose=True)

  @agent
  def sentence_splitter_agent(self):
    return Agent(config=self.agents_config['sentence_splitter_agent'], verbose=True)

  @task
  def finish_sentence_task(self):
    return Task(config=self.tasks_config['finish_sentence_task'], verbose=True)

  @task
  def chat_task(self):
    return Task(config=self.tasks_config['chat_task'], verbose=True)

  @task
  def sentence_split_task(self):
    return Task(config=self.tasks_config['sentence_split_task'], verbose=True)

  @task
  def rearrange_word_task(self):
    return Task(config=self.tasks_config['rearrange_word_task'], verbose=True)

  @crew
  def sgin_gpt_crew(self) -> Crew:
    finisher = self.sentence_finisher_agent()
    chatter  = self.chat_model_agent()
    splitter = self.sentence_splitter_agent()

    t1 = self.finish_sentence_task()
    t2 = self.chat_task()
    t3 = self.sentence_split_task()
    t4 = self.rearrange_word_task()

    t2.context = [t1]
    t3.context = [t2]
    t4.context = [t3]

    return Crew(
      agents=[finisher, chatter, splitter],
      tasks=[t1, t2, t3],  # 필요시 t4 추가
      verbose=True,
    )
