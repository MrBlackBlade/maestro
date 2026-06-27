import json
import matplotlib.pyplot as plt

MOODS = ["angry", "exciting", "fear", "funny", "happy",
         "lazy", "magnificent", "quiet", "romantic", "sad", "warm"]

SELECTED_INDICES = []

INDICES = input(r"""Enter selected mood_ids to display in the graph (separated by spaces):
    0. Angry
    1. Exciting
    2. Fear
    3. Funny
    4. Happy
    5. Lazy
    6. Magnificent
    7. Quiet
    8. Romantic
    9. Sad
    10. Warm
""")

SELECTED_INDICES.extend(INDICES.strip(" ").split())
SELECTED_INDICES = [int(i) for i in SELECTED_INDICES]

print(SELECTED_INDICES)

with open("prob.json") as f:
    prob_dict = json.load(f)

SELECTED_PROBS = {}
for idx in SELECTED_INDICES:
    SELECTED_PROBS[idx] = prob_dict[str(idx)]

plt.figure(figsize=(14, 5))

if not SELECTED_INDICES:
    for mood_id, probs in prob_dict.items():
        plt.plot(probs, label=MOODS[int(mood_id)], linewidth=0.8)
else:
    for mood_id, probs in SELECTED_PROBS.items():
        plt.plot(probs, label=MOODS[int(mood_id)], linewidth=0.8)

plt.xlabel("Generated Token")
plt.ylabel("Mood Probability")
plt.title("Mood Probabilities Over Generation")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.show()