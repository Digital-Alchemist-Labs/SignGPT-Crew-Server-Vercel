# --- app.py 상단: 안전한 경로 설정 ---
from pathlib import Path
import json, os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# api/ 디렉터리
API_DIR = Path(__file__).resolve().parent
DATA_PATH = API_DIR / "data" / "english_words.json"

# .env (로컬용)
load_dotenv()

# ==== 여기 추가: FastAPI 앱 생성 ====
app = FastAPI(
    title="SignGPT Crew Server",
    description="API server for processing ASL tokens using AI agents",
    version="1.0.0",
)

# (선택) CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 프로덕션에서는 도메인 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 로드
try:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        asl_dataset_raw = json.load(f)
    asl_dataset = [asl_dataset_raw[w].upper() for w in asl_dataset_raw]
except FileNotFoundError as exc:
    raise RuntimeError(f"ASL dataset file not found at {DATA_PATH}") from exc

from .crew import SginGPTCrew  # 같은 폴더

# Pydantic models for request/response


class ProcessTokensRequest(BaseModel):
  words: List[str] = Field(
      ...,
      description="List of ASL tokens to process",
      example=["YOU", "NAME", "WHAT"]
  )


class ChatRequest(BaseModel):
  message: str = Field(
      ...,
      description="Direct message/question for the chat model agent",
      example="What is your name?"
  )


class ProcessTokensResponse(BaseModel):
  content: str = Field(description="Final processed result")
  output_history: List[Dict[str, Any]] = Field(
      description="History of outputs from each task")


class ChatOnlyResponse(BaseModel):
  content: str = Field(description="Chat model agent response")
  output_history: List[Dict[str, Any]] = Field(
      description="History of outputs up to chat task")


class ChatAgentResponse(BaseModel):
  content: str = Field(description="Direct chat model agent response")
  agent_info: Dict[str, str] = Field(
      description="Information about the chat agent")


class HealthResponse(BaseModel):
  status: str
  message: str
  asl_dataset_size: int


class ErrorResponse(BaseModel):
  error: str
  detail: Optional[str] = None

# Initialize crew instance


def get_crew_instance():
  """Get a new crew instance"""
  return SginGPTCrew()


@app.get("/", response_model=Dict[str, str])
async def root():
  """Root endpoint with basic information"""
  return {
      "message": "SignGPT Crew Server is running",
      "docs": "/docs",
      "health": "/health"
  }


@app.get("/health", response_model=HealthResponse)
async def health_check():
  """Health check endpoint"""
  # Check if OpenAI API key is configured
  openai_key_configured = bool(os.getenv("OPENAI_API_KEY"))

  return HealthResponse(
      status="healthy" if openai_key_configured else "warning",
      message="Service is running" if openai_key_configured else "Service running but OPENAI_API_KEY not configured",
      asl_dataset_size=len(asl_dataset)
  )


@app.post("/process-tokens", response_model=ProcessTokensResponse)
async def process_tokens(request: ProcessTokensRequest):
  """
  Process ASL tokens through the SignGPT crew workflow

  This endpoint takes a list of ASL tokens and processes them through
  multiple AI agents to generate natural language output.
  Returns only the final result string.
  """
  try:
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
      raise HTTPException(
          status_code=500,
          detail="OPENAI_API_KEY not configured. Please set it in your environment or .env file."
      )

    # Validate input
    if not request.words:
      raise HTTPException(
          status_code=400,
          detail="Words list cannot be empty"
      )

    # Process tokens through the crew
    crew_instance = get_crew_instance()
    crew = crew_instance.sgin_gpt_crew()
    result = crew.kickoff(
        inputs={'words': request.words, 'ASL_dataset': asl_dataset}
    )

    # Extract output history from tasks
    output_history = []
    for i, task in enumerate(crew.tasks):
      if hasattr(task, 'output') and task.output:
        output_history.append({
            'task_index': i,
            'task_description': task.description[:100] + '...' if len(task.description) > 100 else task.description,
            'agent_role': task.agent.role if task.agent else 'Unknown',
            'output': str(task.output)
        })

    # Return result with history
    return ProcessTokensResponse(
        content=str(result),
        output_history=output_history
    )

  except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Error processing tokens: {str(e)}"
    ) from e


@app.post("/process-tokens-chat-only", response_model=ChatOnlyResponse)
async def process_tokens_chat_only(request: ProcessTokensRequest):
  """
  Process ASL tokens through only the first two agents (sentence finisher and chat model)

  This endpoint takes a list of ASL tokens and processes them through:
  1. Sentence finisher agent - converts tokens to natural English
  2. Chat model agent - generates conversational response

  Returns the chat model response and output history of both tasks.
  """
  try:
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
      raise HTTPException(
          status_code=500,
          detail="OPENAI_API_KEY not configured. Please set it in your environment or .env file."
      )

    # Validate input
    if not request.words:
      raise HTTPException(
          status_code=400,
          detail="Words list cannot be empty"
      )

    # Create crew instance and get individual agents/tasks
    crew_instance = get_crew_instance()

    # Get agents
    finisher = crew_instance.sentence_finisher_agent()
    chatter = crew_instance.chat_model_agent()

    # Get tasks
    t1 = crew_instance.finish_sentence_task()
    t2 = crew_instance.chat_task()

    # Set up context chaining: t1 -> t2
    t2.context = [t1]

    # Create a crew with only first two tasks
    from crewai import Crew
    chat_only_crew = Crew(
        agents=[finisher, chatter],
        tasks=[t1, t2],
        verbose=True,
    )

    # Execute the crew
    result = chat_only_crew.kickoff(
        inputs={'words': request.words, 'ASL_dataset': asl_dataset}
    )

    # Extract output history from tasks
    output_history = []
    for i, task in enumerate(chat_only_crew.tasks):
      if hasattr(task, 'output') and task.output:
        output_history.append({
            'task_index': i,
            'task_description': task.description[:100] + '...' if len(task.description) > 100 else task.description,
            'agent_role': task.agent.role if task.agent else 'Unknown',
            'output': str(task.output)
        })

    # Return chat model result with history
    return ChatOnlyResponse(
        content=str(result),
        output_history=output_history
    )

  except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Error processing tokens (chat only): {str(e)}"
    ) from e


@app.post("/chat", response_model=ChatAgentResponse)
async def chat_with_agent(request: ChatRequest):
  """
  Direct chat with the chat model agent only

  This endpoint bypasses all other agents and directly sends the message
  to the chat model agent for a conversational response.
  """
  try:
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
      raise HTTPException(
          status_code=500,
          detail="OPENAI_API_KEY not configured. Please set it in your environment or .env file."
      )

    # Validate input
    if not request.message.strip():
      raise HTTPException(
          status_code=400,
          detail="Message cannot be empty"
      )

    # Create crew instance and get chat agent only
    crew_instance = get_crew_instance()
    chat_agent = crew_instance.chat_model_agent()
    chat_task = crew_instance.chat_task()

    # Create a crew with only the chat agent and task
    from crewai import Crew
    chat_crew = Crew(
        agents=[chat_agent],
        tasks=[chat_task],
        verbose=True,
    )

    # Execute the crew with direct message input
    result = chat_crew.kickoff(
        inputs={'message': request.message, 'ASL_dataset': ''}
    )

    # Return direct chat response
    return ChatAgentResponse(
        content=str(result),
        agent_info={
            'agent_role': chat_agent.role,
            'agent_goal': chat_agent.goal,
            'model': 'gpt-4o-mini'
        }
    )

  except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Error in chat: {str(e)}"
    ) from e


@app.get("/asl-dataset", response_model=Dict[str, Any])
async def get_asl_dataset():
  """Get information about the available ASL dataset"""
  return {
      "total_words": len(asl_dataset),
      "sample_words": asl_dataset[:20],  # First 20 words as sample
      "description": "Available ASL vocabulary tokens"
  }


@app.post("/validate-tokens", response_model=Dict[str, Any])
async def validate_tokens(tokens: List[str]):
  """
  Validate if given tokens exist in the ASL dataset
  """
  validation_results = {}
  valid_tokens = []
  invalid_tokens = []

  for token in tokens:
    token_upper = token.upper()
    if token_upper in asl_dataset:
      valid_tokens.append(token_upper)
      validation_results[token] = {
          "valid": True, "normalized": token_upper}
    else:
      invalid_tokens.append(token)
      validation_results[token] = {"valid": False, "normalized": None}

  return {
      "validation_results": validation_results,
      "summary": {
          "total_tokens": len(tokens),
          "valid_count": len(valid_tokens),
          "invalid_count": len(invalid_tokens),
          "valid_tokens": valid_tokens,
          "invalid_tokens": invalid_tokens
      }
  }

# Error handlers


@app.exception_handler(404)
async def not_found_handler(request, exc):  # noqa: ARG001
  return {"error": "Not found", "detail": "The requested resource was not found"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):  # noqa: ARG001
  return {"error": "Internal server error", "detail": "An unexpected error occurred"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
