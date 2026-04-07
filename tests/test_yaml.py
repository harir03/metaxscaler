import yaml

with open("openenv.yaml") as f:
    d = yaml.safe_load(f)

print(f"{d['name']} v{d['version']}")
for t in d["tasks"]:
    print(f"  {t['name']} -> {t['grader']}")
assert len(d["tasks"]) >= 3
print("ok")
