import json, glob, os, random, sys
src = sys.argv[1]
out = os.path.dirname(os.path.abspath(__file__))
items = []
for f in sorted(glob.glob(os.path.join(src, "*.json"))):
    model = os.path.basename(f)[:-5]
    for k, g in enumerate(json.load(open(f))):
        items.append({"model": model, "k": k, "q": g["q"], "a": g["a"][:1500]})
random.Random(42).shuffle(items)
key = {}
for i, it in enumerate(items):
    it["id"] = f"r{i:04d}"
    key[it["id"]] = {"model": it["model"], "k": it["k"]}
json.dump(key, open(os.path.join(out, "key.json"), "w"))
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
per = (len(items) + NB - 1) // NB
for b in range(NB):
    chunk = [{"id": it["id"], "question": it["q"], "answer": it["a"]} for it in items[b * per:(b + 1) * per]]
    json.dump(chunk, open(os.path.join(out, f"batch_{b}.json"), "w"), indent=0, ensure_ascii=False)
print(len(items), "items in", NB, "batches of", per)
