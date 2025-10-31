from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
app = FastAPI(
    title="To-Do API",
    description="Simple in-memory to-do API built with FastAPI",
    version="1.0.0"
)
class TodoIn(BaseModel):
    task: str

class TodoOut(BaseModel):
    id: int
    task: str

todos: List[TodoOut] = [
    TodoOut(id=1, task="Buy groceries"),
    TodoOut(id=2, task="Finish assignment"),
]

@app.get("/todos", response_model=List[TodoOut])
def get_todos():
    """Return all to-do items"""
    return todos

@app.post("/todos", response_model=TodoOut, status_code=201)
def create_todo(todo: TodoIn):
    new_id = max((t.id for t in todos), default=0) + 1
    new_todo = TodoOut(id=new_id, task=todo.task)
    todos.append(new_todo)
    return new_todo
