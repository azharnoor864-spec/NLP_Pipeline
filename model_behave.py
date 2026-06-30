# ============================================================
# DAY 9 — System/User/Assistant Roles, Structured JSON, Few-Shot, Personas
# Run in Google Colab. Needs GROQ_API_KEY in userdata (Secrets tab).
# ============================================================

from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()
client = Groq()
# MODEL = "llama-3.3-70b-versatile"

def call_model(messages,temperature):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


# ============================================================
# 1. STRICT JSON SCHEMA — math derivation use case
# ============================================================
# Schema: given a math problem, model must show step-by-step
# derivation AND a final structured answer.

SYSTEM_JSON_MATH = """You are a math derivation engine.
Respond ONLY with valid JSON, no markdown fences, no extra text.
Schema:
{
  "problem": string,
  "steps": [string],       // each step of the derivation, in order
  "final_answer": string,
  "method": string         // e.g. "integration by parts", "substitution"
}
Never skip steps. Never add fields outside this schema."""

def solve_math_structured(problem):
    messages = [
        {"role": "system", "content": SYSTEM_JSON_MATH},
        {"role": "user", "content": problem}
    ]
    raw = call_model(messages, temperature=0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": "That wasn't valid JSON. Return ONLY the JSON object."})
        raw_retry = call_model(messages, temperature=0)
        try:
            return json.loads(raw_retry)
        except json.JSONDecodeError:
            return {"error": "failed to produce valid JSON", "raw_output": raw_retry}


# ============================================================
# 2. FEW-SHOT EXAMPLES — teaching the integration pattern
# ============================================================
# Three worked examples shown as fake user/assistant turns,
# then the real question is asked.

few_shot_math_messages = [
    {"role": "system", "content": SYSTEM_JSON_MATH},

    # Example 1
    {"role": "user", "content": "Integrate x dx"},
    {"role": "assistant", "content": json.dumps({
        "problem": "∫ x dx",
        "steps": [
            "Apply the power rule: ∫ x^n dx = x^(n+1)/(n+1) + C",
            "Here n = 1, so result = x^2/2 + C"
        ],
        "final_answer": "x^2/2 + C",
        "method": "power rule"
    })},

    # Example 2
    {"role": "user", "content": "Integrate x * e^x dx"},
    {"role": "assistant", "content": json.dumps({
        "problem": "∫ x e^x dx",
        "steps": [
            "Use integration by parts: ∫u dv = uv - ∫v du",
            "Let u = x, dv = e^x dx, so du = dx, v = e^x",
            "Apply formula: x*e^x - ∫e^x dx",
            "Integrate remaining term: x*e^x - e^x + C"
        ],
        "final_answer": "x*e^x - e^x + C",
        "method": "integration by parts"
    })},

    # Example 3
    {"role": "user", "content": "Integrate 2x * cos(x^2) dx"},
    {"role": "assistant", "content": json.dumps({
        "problem": "∫ 2x cos(x^2) dx",
        "steps": [
            "Use substitution: let t = x^2, so dt = 2x dx",
            "Rewrite integral as ∫ cos(t) dt",
            "Integrate: sin(t) + C",
            "Substitute back t = x^2: sin(x^2) + C"
        ],
        "final_answer": "sin(x^2) + C",
        "method": "substitution"
    })},

    # Real question — model should now follow the same JSON + step pattern
    {"role": "user", "content": "Integrate x^2 * ln(x) dx"}
]

def run_few_shot_demo():
    raw = call_model(few_shot_math_messages, temperature=0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid JSON", "raw_output": raw}


# ============================================================
# 3. PRODUCTION PROMPTS — 4 tasks
# ============================================================

PROMPTS = {

    # (a) Structured JSON generation
    "json_generation": {
        "system": SYSTEM_JSON_MATH,
        "temperature": 0
    },

    # (b) Unstructured text parsing — free text in, free text out
    "unstructured": {
        "system": """You are a free-form writing assistant.
Respond in plain natural text. No JSON, no structure required.
Just write naturally based on what the user asks.""",
        "temperature": 0.7
    },

    # (c) Code generation — short LLM inference snippet
    "code_generation": {
        "system": """You are a senior Python engineer.
Write clean, minimal, working code only.
No explanations unless explicitly asked. No unnecessary comments.""",
        "temperature": 0.2
    },

    # (d) Document summarization — NLP topic
    "summarization": {
        "system": """Summarize the given text in 3-5 sentences.
Preserve key facts and technical accuracy. No opinions, no fluff.""",
        "temperature": 0.4
    }
}

def run_task(task_name, user_input):
    config = PROMPTS[task_name]
    messages = [
        {"role": "system", "content": config["system"]},
        {"role": "user", "content": user_input}
    ]
    return call_model(messages, temperature=config["temperature"])


# ============================================================
# 4. PERSONA SWITCHING — formal, casual, technical
# ============================================================

PERSONAS = {
    "formal": "You are a formal corporate assistant. Write professional, polite, complete-sentence language suitable for official correspondence.",
    "casual": "You're a warm, relaxed assistant. Use casual, friendly, everyday language as if texting someone you care about.",
    "technical": "You are a technical/math expert. Be precise, use correct terminology, show clear step-by-step reasoning."
}

def ask_with_persona(persona, user_input, temperature=0.5):
    messages = [
        {"role": "system", "content": PERSONAS[persona]},
        {"role": "user", "content": user_input}
    ]
    return call_model(messages, temperature=temperature)


# ============================================================
# 5. RUN EVERYTHING — demo calls
# ============================================================

if __name__ == "__main__":

    print("===== 1. FEW-SHOT MATH DERIVATION (∫ x^2 ln(x) dx) =====")
    print(json.dumps(run_few_shot_demo(), indent=2))

    print("\n===== 2. JSON GENERATION (single problem, no few-shot) =====")
    print(json.dumps(solve_math_structured("Integrate sin(x) * cos(x) dx"), indent=2))

    print("\n===== 3. UNSTRUCTURED TEXT (free write) =====")
    print(run_task("unstructured", "Write a short note about why I love rainy days."))

    print("\n===== 4. CODE GENERATION (short LLM inference snippet) =====")
    print(run_task("code_generation",
        "Write a short Python function that calls an LLM API and returns just the text response."))

    print("\n===== 5. SUMMARIZATION (NLP topic) =====")
    nlp_text = """Named Entity Recognition (NER) is a subtask of NLP that identifies and classifies
    key entities in text into predefined categories such as person names, organizations, locations,
    dates, and monetary values. Modern NER systems often use transformer-based models like BERT,
    fine-tuned on labeled datasets such as CoNLL-2003. NER is widely used in information extraction,
    question answering, and search engines to convert unstructured text into structured, searchable data."""
    print(run_task("summarization", nlp_text))

    print("\n===== 6. PERSONA: FORMAL — letter to a company =====")
    print(ask_with_persona("formal",
        "Write a letter to a software company requesting an internship in their AI/ML department, mentioning my final year project EBQR."))

    print("\n===== 7. PERSONA: CASUAL — message to a friend/family =====")
    print(ask_with_persona("casual",
        "Write a short message to my friend telling them I finished Day 9 of my internship tasks and want to hang out this weekend."))

    print("\n===== 8. PERSONA: TECHNICAL — solve a problem =====")
    print(ask_with_persona("technical",
        "Solve: find the derivative of f(x) = x^3 * e^(2x), showing each step."))