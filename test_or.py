import os, json, urllib.request, urllib.error
from dotenv import load_dotenv

print(".env in project root:", os.path.exists(".env"))
load_dotenv(".env", override=True)
key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free")
print("key loaded:", bool(key), "| starts with sk-or-:", key.startswith("sk-or-"), "| length:", len(key))
print("model:", model)

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps({"model": model, "max_tokens": 20,
                     "messages": [{"role": "user", "content": "say hi"}]}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("RESULT OK:", json.loads(r.read())["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print("RESULT HTTP", e.code, e.read().decode()[:400])
except Exception as e:
    print("RESULT ERROR:", e)

print("\nFree models right now:")
try:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        for m in json.loads(r.read())["data"]:
            p = m.get("pricing", {})
            if str(p.get("prompt")) == "0" and str(p.get("completion")) == "0":
                print(" ", m["id"])
except Exception as e:
    print("could not list models:", e)
