import json, random
al = json.load(open("data/alpaca_cleaned.json", encoding="utf-8")); random.Random(5).shuffle(al)
with open("data/alpaca.jsonl", "w", encoding="utf-8") as f:
    for r in al[-500:]:
        q = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": r["output"]}]}) + "\n")
print("ok")
