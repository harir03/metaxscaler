import sys
sys.path.insert(0, ".")
from server.app import app

print(f"app: {app.title} ({len(app.routes)} routes)")
for r in app.routes:
    m = getattr(r, "methods", "")
    print(f"  {r.path} {m}")
print("ok")
