import ast
files = ["app/main.py", "app/api/routes.py", "app/models/schemas.py"]
for f in files:
    ast.parse(open(f).read())
    print(f"  ✅ {f}")
print("All files parse OK")
