import os, json, time, urllib.request, urllib.error
from dotenv import load_dotenv
load_dotenv(".env", override=True)
key = (os.getenv("OPENROUTER_API_KEY") or "").strip()

candidates = [
    "poolside/laguna-s-2.1:free",
    "cohere/north-mini-code:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
]

for model in candidates:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "max_tokens": 30,
                         "messages": [{"role": "user", "content": "Write a Python function that adds two numbers."}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            json.loads(r.read())
            print(f"OK    {time.time()-start:5.1f}s  {model}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} {time.time()-start:5.1f}s  {model}")
    except Exception as e:
        print(f"ERR   {time.time()-start:5.1f}s  {model}  {e}")
