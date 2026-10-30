"""Small calibration / evaluation corpora (calib and eval are disjoint)."""
import glob, os
from . import chat
from .config import MODEL_DIR, HERE

FR = [
    "La photosynthèse est le processus par lequel les plantes, les algues et certaines bactéries convertissent l'énergie lumineuse en énergie chimique. "
    "Elle se déroule dans les chloroplastes, où la chlorophylle absorbe principalement la lumière rouge et bleue. L'eau et le dioxyde de carbone sont transformés en glucose et en dioxygène.",
    "Pour préparer une bonne pâte à crêpes, mélangez 250 g de farine, 4 œufs, un demi-litre de lait et une pincée de sel. Laissez reposer la pâte une heure, "
    "puis faites cuire de fines couches dans une poêle chaude légèrement beurrée. Retournez-les lorsque les bords se détachent.",
    "Le projet de loi sera examiné la semaine prochaine par la commission des affaires économiques. Selon le ministre, il vise à simplifier les démarches administratives des petites entreprises "
    "tout en renforçant la protection des données personnelles des clients. Les syndicats ont exprimé leurs réserves.",
    "Il était une fois, dans un village niché au creux des montagnes, un vieil horloger qui réparait le temps. Chaque matin, il ouvrait sa boutique, "
    "accrochait son tablier gris et regardait les aiguilles tourner. Les enfants venaient parfois lui demander l'heure, et il leur répondait toujours par une histoire.",
]
EN = [
    "A hash table stores key-value pairs in an array indexed by a hash of the key. Collisions are resolved either by chaining, where each bucket holds a linked list, "
    "or by open addressing, where the algorithm probes other slots. The average cost of lookup, insertion and deletion is O(1), but the worst case is O(n).",
    "Interest rates are one of the main tools central banks use to control inflation. Raising rates makes borrowing more expensive, which cools demand, while lowering them "
    "stimulates investment and spending. The effect usually takes months to appear in the real economy.",
    "To reset your password, open the settings page, choose Security, and click on Change password. You will receive an email with a verification code that expires after ten minutes. "
    "If you do not see the message, check your spam folder or request a new code.",
]
HELD_OUT = [
    "The mitochondrion is an organelle found in most eukaryotic cells. Often called the powerhouse of the cell, it generates most of the cell's supply of adenosine triphosphate, "
    "used as a source of chemical energy. Mitochondria have their own small circular genome inherited maternally.",
    "Le TGV est un train à grande vitesse exploité par la SNCF. Mis en service en 1981 entre Paris et Lyon, il a progressivement relié la plupart des grandes villes françaises "
    "et plusieurs capitales européennes. Sa vitesse commerciale maximale atteint 320 kilomètres par heure sur certaines lignes.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint([fibonacci(i) for i in range(10)])",
    "Dear team, the quarterly review is scheduled for Thursday at 10 am in the main conference room. Please send your slides by Wednesday evening and make sure "
    "to include the updated figures for the European market. Thank you for your hard work this quarter.",
]


def calibration_texts(max_tokens=1100):
    texts = []
    readme = open(f"{MODEL_DIR}/README.md", encoding="utf-8").read()
    # prose / tables from the model card in chunks
    chunk = 3500
    for i in range(0, min(len(readme), 5 * chunk), chunk):
        texts.append(readme[i:i + chunk])
    for f in sorted(glob.glob(os.path.join(HERE, "lfm3npu", "*.py")))[:6]:
        texts.append(open(f, encoding="utf-8").read()[:3500])
    texts += FR + EN
    # chat-formatted conversations (system + user + assistant, with and without tools)
    from .tools import make_registry, Notes
    schemas, _ = make_registry(Notes())
    convs = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explique-moi la différence entre un processus et un thread."},
         {"role": "assistant", "content": "Un processus possède son propre espace mémoire, alors qu'un thread partage l'espace mémoire du processus qui l'a créé. Les threads sont plus légers à créer et à commuter."}],
        [{"role": "system", "content": "You are a helpful assistant with access to tools."}, {"role": "user", "content": "What is the weather in Paris and what is 12*7?"},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "0", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                                                          {"id": "1", "type": "function", "function": {"name": "calculator", "arguments": {"expression": "12*7"}}}]},
         {"role": "tool", "tool_call_id": "0", "content": '{"city": "Paris", "temperature": "18C", "conditions": "sunny"}'}, {"role": "tool", "tool_call_id": "1", "content": "84"},
         {"role": "assistant", "content": "It is 18C and sunny in Paris, and 12*7 equals 84."}],
        [{"role": "system", "content": "You are a helpful assistant with access to tools."}, {"role": "user", "content": "Sauvegarde une note 'courses' avec 'lait, oeufs, pain' puis liste mes notes."},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "0", "type": "function", "function": {"name": "save_note", "arguments": {"title": "courses", "content": "lait, oeufs, pain"}}}]},
         {"role": "tool", "tool_call_id": "0", "content": "saved note courses"},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "list_notes", "arguments": {}}}]},
         {"role": "tool", "tool_call_id": "1", "content": '["courses"]'},
         {"role": "assistant", "content": "Votre note « courses » est enregistrée. Vous avez 1 note : courses."}],
    ]
    for c in convs:
        texts.append(chat.render(c, schemas, add_generation_prompt=False))
    out = []
    for t in texts:
        ids = chat.encode(t)[:max_tokens]
        if len(ids) > 40:
            out.append(ids)
    return out


def eval_texts(max_tokens=300):
    out = []
    for t in HELD_OUT:
        ids = chat.encode(t)[:max_tokens]
        out.append(ids)
    return out
