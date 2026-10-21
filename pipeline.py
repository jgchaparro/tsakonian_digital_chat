"""One local translator for both directions, with Greek conversation on Ollama or NIM."""

import httpx

DEFAULT_NIM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
KRIKRI_MODEL = "hf.co/ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M"

TRANSLATION_SYSTEM = (
    "You are a precise translation assistant. Output only the translation, nothing else."
)
GREEK_SYSTEM = (
    "Είσαι ο Λεωνίδας, ένας ψηφιακός βοηθός για συνομιλία στα ελληνικά και στα τσακώνικα. "
    "Είσαι εδώ για να βοηθάς τον χρήστη να μιλάει τσακώνικα. Ο χρήστης γράφει στα "
    "τσακώνικα· τα μηνύματά του σού φτάνουν μεταφρασμένα στα ελληνικά και η απάντησή "
    "σου θα μεταφραστεί στα τσακώνικα, οπότε αν κάτι δεν βγάζει νόημα, ζήτα ευγενικά "
    "διευκρίνιση. Απαντάς ΠΑΝΤΑ στα ελληνικά, με σύντομες και απλές "
    "προτάσεις και απλό λεξιλόγιο, χωρίς emoji, markdown, λίστες ή πίνακες. "
    "Δώσε μόνο την τελική απάντηση, σε 1–3 προτάσεις και έως 60 λέξεις."
)


def translate(text, origin, target, *, model="leonida", base_url="http://localhost:11434",
              keep_alive="1h"):
    if (origin, target) not in {("Tsakonian", "Greek"), ("Greek", "Tsakonian")}:
        raise ValueError("Translation must be between Greek and Tsakonian.")
    # Accept the previous app's OpenAI-compatible URL as well as the native URL.
    base_url = base_url.rstrip("/").removesuffix("/v1")
    user_prompt = f"Translate from {origin} to {target}:\n\n{text}"
    # The installed GGUF embeds a fixed-direction Jinja template. raw=True
    # bypasses it; serialize the original tokenizer's no-tools Llama 3.1 format.
    # The date prelude is the tokenizer's training default, not today's date.
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
        f"{TRANSLATION_SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    response = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        },
        timeout=httpx.Timeout(180, connect=10),
    )
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        raise ValueError(f"Ollama: {result['error']}")
    if result.get("done_reason") == "length":
        raise ValueError("Leonida reached its output limit. Try a shorter message.")
    content = result["response"].strip()
    if not content:
        raise ValueError("Leonida returned an empty translation.")
    return content


def list_ollama_models(base_url="http://localhost:11434", *, exclude=("leonida",)):
    """Installed Ollama chat models, without the translator; empty if Ollama is down."""
    base_url = base_url.rstrip("/").removesuffix("/v1")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=httpx.Timeout(5, connect=2))
        response.raise_for_status()
        names = [model["name"] for model in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return []
    return [name for name in names
            if not any(name.split(":")[0].startswith(prefix) for prefix in exclude)]


def parse_model_list(value, default):
    """Comma-separated model names, keeping order and the default first."""
    models = [name.strip() for name in (value or "").split(",") if name.strip()]
    return list(dict.fromkeys([default, *models]))


def greek_messages(history, greek_message):
    messages = [{"role": "system", "content": GREEK_SYSTEM}]
    for turn in history[-10:]:
        messages.append({"role": "user", "content": turn["ell_in"]})
        messages.append({"role": "assistant", "content": turn["ell_out"]})
    messages.append({"role": "user", "content": greek_message})
    return messages


def ask_ollama(history, greek_message, *, model=DEFAULT_OLLAMA_MODEL,
               base_url="http://localhost:11434", keep_alive="5m"):
    base_url = base_url.rstrip("/").removesuffix("/v1")
    payload = {
        "model": model, "messages": greek_messages(history, greek_message),
        "stream": False, "keep_alive": keep_alive,
        "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": 512},
    }
    if model.split(":")[0] == "gemma4":
        # Short conversation replies do not need a separate thinking pass.
        payload["think"] = False
    response = httpx.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=httpx.Timeout(180, connect=10),
    )
    if response.status_code == 404:
        raise ValueError(f"Local model {model} is not installed. See the local setup instructions in README.md.")
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        raise ValueError(f"Ollama: {result['error']}")
    if result.get("done_reason") == "length":
        raise ValueError("The local assistant could not finish its reply. Please retry this step.")
    content = (result["message"]["content"] or "").strip()
    if not content:
        raise ValueError("The local assistant returned an empty Greek reply.")
    return content


def ask_nim(client, model, history, greek_message, *, max_tokens=2048, reasoning_effort="low"):
    if not 1 <= max_tokens <= 4096:
        raise ValueError("NIM_MAX_TOKENS must be between 1 and 4096.")
    messages = greek_messages(history, greek_message)
    options = {}
    if model.startswith("openai/gpt-oss-"):
        # Its generation budget includes hidden reasoning, even for short replies.
        options["reasoning_effort"] = reasoning_effort
    # Retry only a token-limited generation, once, with more room for reasoning.
    # Never pass an unfinished answer to the translator or add it to history.
    budgets = [max_tokens]
    if max_tokens < 4096:
        budgets.append(min(max_tokens * 2, 4096))
    for budget in budgets:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.4, max_tokens=budget, **options,
        )
        choice = response.choices[0]
        if choice.finish_reason != "length":
            break
    else:
        raise ValueError("The assistant could not finish its reply. Please retry this step.")
    content = (choice.message.content or "").strip()
    if not content:
        raise ValueError("NIM returned an empty Greek reply.")
    return content
