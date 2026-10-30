"""Chat template rendering (jinja2, same helpers as transformers), sampling, tool-call parsing."""
import json, re, datetime
import numpy as np
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from .config import MODEL_DIR

from jinja2 import nodes
from jinja2.ext import Extension


class _Generation(Extension):
    """HF `{% generation %}...{% endgeneration %}` marks assistant spans for training masks: transparent here."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_pass"), [], [], body).set_lineno(lineno)

    def _pass(self, caller):
        return caller()


_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_Generation])
_env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
_env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.exceptions.TemplateError(m))
_env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
_tpl = _env.from_string(open(f"{MODEL_DIR}/chat_template.jinja", encoding="utf-8").read())
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
EOS = 124900          # <|im_end|>
THINK_END = None


def render(messages, tools=None, add_generation_prompt=True, **kw):
    return _tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                       bos_token="<|startoftext|>", eos_token="<|im_end|>", **kw)


def encode(text):
    return tok.encode(text, add_special_tokens=False).ids


def decode(ids, skip_special=False):
    return tok.decode(list(ids), skip_special_tokens=skip_special)


def sample(logits, temperature=0.1, top_k=50, top_p=1.0, rep_penalty=1.1, history=(), rng=None):
    lg = logits.astype(np.float64).copy()
    if rep_penalty != 1.0 and len(history):
        idx = np.unique(np.asarray(history[-512:]))
        v = lg[idx]
        lg[idx] = np.where(v > 0, v / rep_penalty, v * rep_penalty)
    if temperature <= 0:
        return int(np.argmax(lg))
    lg /= temperature
    if top_k and top_k < lg.size:
        kth = np.partition(lg, -top_k)[-top_k]
        lg[lg < kth] = -np.inf
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    if top_p < 1.0:
        o = np.argsort(-p); c = np.cumsum(p[o]); keep = o[: int(np.searchsorted(c, top_p) + 1)]
        m = np.zeros_like(p); m[keep] = p[keep]; p = m / m.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(p.size, p=p))


_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_tool_calls(text):
    """Parse LFM3 XML tool calls -> [{'name':..., 'arguments': {...}}]; values JSON-decoded when possible."""
    calls = []
    for name, body in _CALL.findall(text):
        args = {}
        for k, v in _PARAM.findall(body):
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        calls.append({"name": name, "arguments": args})
    return calls
