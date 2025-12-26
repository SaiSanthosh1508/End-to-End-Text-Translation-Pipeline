import os

print("--- Checking 'ultralytics' folder structure ---")
if not os.path.exists("ultralytics"):
    print("❌ ERROR: 'ultralytics' folder not found!")
else:
    contents = os.listdir("ultralytics")
    if "ultralytics" in contents and os.path.isdir("ultralytics/ultralytics"):
        print("⚠️ PROBLEM FOUND: Nested folders detected.")
        print("   You have app/ultralytics/ultralytics/...")
        print("   You need to move the inner folder OUT.")
    elif "__init__.py" in contents:
        print("✅ STRUCTURE LOOKS GOOD.")
        print("   Checking if YOLO is defined in __init__.py...")
        with open("ultralytics/__init__.py", "r", encoding="utf-8") as f:
            if "class YOLO" in f.read() or "from .models import YOLO" in f.read():
                print("   ✅ YOLO class found in __init__.py")
            else:
                print("   ❌ YOLO class NOT found in __init__.py. Did you edit it?")
    else:
        print("❌ ERROR: No '__init__.py' found. This is not a valid Python package.")
        print("   Current contents:", contents)