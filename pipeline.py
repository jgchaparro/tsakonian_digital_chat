"""One local translator for both directions, with Greek conversation on Ollama or NIM."""

import httpx

DEFAULT_NIM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
KRIKRI_MODEL = "hf.co/ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M"

# Greek-reply providers reached with an API key through the OpenAI SDK.
# Keys come only from environment variables; models can be extended in <models_env>.
PROVIDERS = {
    "nim": {"label": "NVIDIA NIM", "key_envs": ("NVIDIA_API_KEY",),
            "base_url": "https://integrate.api.nvidia.com/v1",
            "default_model": DEFAULT_NIM_MODEL, "models_env": "NIM_MODELS"},
    "openai": {"label": "OpenAI", "key_envs": ("OPENAI_API_KEY",), "base_url": None,
               "default_model": "gpt-5-mini", "models_env": "OPENAI_MODELS"},
    "gemini": {"label": "Google Gemini", "key_envs": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "default_model": "gemini-2.5-flash", "models_env": "GEMINI_MODELS"},
}


def provider_key(provider, environ):
    """The API key of a provider from the environment, or None."""
    for name in PROVIDERS[provider]["key_envs"]:
        if environ.get(name):
            return environ[name]
    return None

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
              keep_alive="1h", endpoint_url=None, endpoint_token=None):
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
    if endpoint_url:
        return _translate_endpoint(prompt, endpoint_url, endpoint_token)
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


def _translate_endpoint(prompt, endpoint_url, token):
    """Same raw prompt on a hosted OpenAI-style /v1/completions server (llama.cpp, TGI, vLLM)."""
    url = endpoint_url.rstrip("/").removesuffix("/v1") + "/v1/completions"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.post(
        url, headers=headers,
        json={"prompt": prompt, "temperature": 0, "max_tokens": 512, "stop": ["<|eot_id|>"]},
        timeout=httpx.Timeout(180, connect=30),
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Leonida reached its output limit. Try a shorter message.")
    content = (choice.get("text") or "").strip()
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


def _is_openai_reasoning(model):
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def ask_openai_compatible(client, model, history, greek_message, *, provider="nim",
                          max_tokens=2048, reasoning_effort="low"):
    """Greek reply from NIM, OpenAI or Gemini (all through the OpenAI SDK)."""
    if not 1 <= max_tokens <= 4096:
        raise ValueError("NIM_MAX_TOKENS must be between 1 and 4096.")
    label = PROVIDERS[provider]["label"]
    messages = greek_messages(history, greek_message)
    options = {}
    if provider == "openai":
        reasoning = _is_openai_reasoning(model)
        # Reasoning models reject a custom temperature and the max_tokens name.
        if not reasoning:
            options["temperature"] = 0.4
        if reasoning:
            options["reasoning_effort"] = "minimal" if model.startswith("gpt-5") else reasoning_effort
    else:
        options["temperature"] = 0.4
        # Hidden reasoning shares the generation budget on these models.
        if provider == "gemini" or model.startswith("openai/gpt-oss-"):
            options["reasoning_effort"] = reasoning_effort
    token_option = "max_completion_tokens" if provider == "openai" else "max_tokens"
    # Retry only a token-limited generation, once, with more room for reasoning.
    # Never pass an unfinished answer to the translator or add it to history.
    budgets = [max_tokens]
    if max_tokens < 4096:
        budgets.append(min(max_tokens * 2, 4096))
    for budget in budgets:
        response = client.chat.completions.create(
            model=model, messages=messages, **{token_option: budget}, **options,
        )
        choice = response.choices[0]
        if choice.finish_reason != "length":
            break
    else:
        raise ValueError("The assistant could not finish its reply. Please retry this step.")
    content = (choice.message.content or "").strip()
    if not content:
        raise ValueError(f"{label} returned an empty Greek reply.")
    return content


def ask_nim(client, model, history, greek_message, *, max_tokens=2048, reasoning_effort="low"):
    return ask_openai_compatible(client, model, history, greek_message, provider="nim",
                                 max_tokens=max_tokens, reasoning_effort=reasoning_effort)
