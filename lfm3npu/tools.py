"""Sandboxed tools for the agent benchmark."""
import ast, json, math, operator as op, datetime

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
        ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
_FUN = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "log": math.log, "abs": abs, "round": round}


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN:
        return _FUN[n.func.id](*[_ev(a) for a in n.args])
    if isinstance(n, ast.Name) and n.id in ("pi", "e"):
        return getattr(math, n.id)
    raise ValueError("unsupported expression")


def calculator(expression: str):
    return str(_ev(ast.parse(expression, mode="eval").body))


_WEATHER = {"paris": ("18C", "sunny"), "london": ("12C", "rainy"), "tokyo": ("22C", "cloudy"),
            "new york": ("15C", "windy"), "lyon": ("17C", "cloudy")}


def get_weather(city: str):
    t = _WEATHER.get(city.strip().lower())
    return json.dumps({"city": city, "temperature": t[0], "conditions": t[1]}) if t else json.dumps({"error": "unknown city"})


def get_time(timezone: str = "UTC"):
    return datetime.datetime(2026, 10, 30, 14, 5, 0).isoformat() + f" ({timezone})"


class Notes:
    def __init__(self):
        self.items = {}

    def save_note(self, title: str, content: str):
        self.items[title] = content
        return f"saved note {title}"

    def read_note(self, title: str):
        return self.items.get(title, "note not found")

    def list_notes(self):
        return json.dumps(sorted(self.items))


def make_registry(notes: Notes):
    def S(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": props, "required": req}}}
    schemas = [
        S("calculator", "Evaluate an arithmetic expression (+ - * / ** % and sqrt, sin, cos, log, pi).",
          {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, ["expression"]),
        S("get_weather", "Get the current weather of a city.", {"city": {"type": "string"}}, ["city"]),
        S("get_time", "Get the current date and time.", {"timezone": {"type": "string"}}, []),
        S("save_note", "Save a note.", {"title": {"type": "string"}, "content": {"type": "string"}}, ["title", "content"]),
        S("read_note", "Read a saved note by title.", {"title": {"type": "string"}}, ["title"]),
        S("list_notes", "List titles of all saved notes.", {}, []),
    ]
    fns = {"calculator": calculator, "get_weather": get_weather, "get_time": get_time, "save_note": notes.save_note,
           "read_note": notes.read_note, "list_notes": notes.list_notes}
    return schemas, fns
