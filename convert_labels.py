import json

with open("labels.json", "r") as f:
    data = json.load(f)

all_labels = data["letters"] + data["words"]
label_map = {label: i for i, label in enumerate(all_labels)}

with open("labels_fixed.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("Converted to labels_fixed.json")