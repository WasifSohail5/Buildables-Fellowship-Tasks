from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Secure To-Do API")

API_KEY = "secret"
API_KEY_HEADER = "X-API-Key"

class TodoIn(BaseModel):
    task: str

class TodoOut(BaseModel):
    id: int
    task: str

todos: List[TodoOut] = [
    TodoOut(id=1, task="Buy groceries"),
    TodoOut(id=2, task="Finish assignment"),
]

def check_api_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/todos", response_model=List[TodoOut])
def get_todos():
    return todos

@app.post("/todos", response_model=TodoOut, status_code=201)
def create_todo(todo: TodoIn, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    new_id = max([t.id for t in todos], default=0) + 1
    new_todo = TodoOut(id=new_id, task=todo.task)
    todos.append(new_todo)
    return new_todo

@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    for i, t in enumerate(todos):
        if t.id == todo_id:
            del todos[i]
            return
    raise HTTPException(status_code=404, detail="Todo not found")
