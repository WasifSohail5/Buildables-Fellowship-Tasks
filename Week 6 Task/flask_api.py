from flask import Flask, request, jsonify

app = Flask(__name__)
todos = [
    {"id": 1, "task": "Study Hard"},
    {"id": 2, "task": "Go to sleep hehehehe"},
]
@app.route("/todos", methods=["GET"])
def get_todos():
    return jsonify(todos), 200

@app.route("/todos", methods=["POST"])
def create_todo():
    data = request.get_json()
    if not data or "task" not in data:
        return jsonify({"error": "task field is required"}), 400

    new_id = max([t["id"] for t in todos], default=0) + 1
    todo = {"id": new_id, "task": data["task"]}
    todos.append(todo)
    return jsonify(todo), 201

if __name__ == "__main__":
    app.run(debug=True)
