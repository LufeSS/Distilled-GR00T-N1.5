# estimate_gr00t_size.py
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
base = "datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

total = 0
for path in fs.find(base):
    if path.endswith("/"):
        continue
    try:
        info = fs.info(path)
        total += info.get("size", 0)
    except Exception:
        pass

print(f"{total} bytes")
print(f"{total/1024/1024/1024:.2f} GB")