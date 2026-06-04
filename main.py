import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================================================================
# STEP 1: DEFINE THE MICRO-EMBEDDING SPACE
# =====================================================================
# We map words to a 3D coordinate system where each axis has a real meaning:
# X-Axis: Food/Biology | Y-Axis: Nanotechnology/Physics | Z-Axis: Color Intensity (Yellow)

vocab = {
    "banana": 0,
    "yellow": 1,
    "nano":   2,
    "fruit":  3,
    "tech":   4
}

# Hand-crafted Static Embeddings (Matrix E)
E = np.array([
    [0.90, 0.15, 0.85],  # banana: High food, low tech, high yellow
    [0.10, 0.00, 0.95],  # yellow: No food, no tech, pure yellow
    [0.00, 0.95, 0.10],  # nano:   No food, extreme tech, low yellow
    [0.95, 0.00, 0.25],  # fruit:  Extreme food, no tech, generic color
    [0.05, 0.90, 0.15]   # tech:   Low food, extreme tech, generic color
])

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# =====================================================================
# STEP 2: THE SELF-ATTENTION MECHANISM (THE CONTEXT UPDATE)
# =====================================================================
def run_contextual_attention(sentence_tokens):
    token_indices = [vocab[token] for token in sentence_tokens]
    vectors = E[token_indices]  # Extract baseline static coordinates
    
    # In a full Transformer: Q = X@W_Q, K = X@W_K, V = X@W_V
    # We use raw vectors directly to keep the geometric dot-product visible.
    Queries = vectors
    Keys = vectors
    Values = vectors
    
    # Compute Raw Dot-Product Alignment Scores
    scores = np.dot(Queries, Keys.T)
    
    # Statistical Scaling: Softmax to turn scores into clean probabilities
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # The Update: Words pull their neighbors' values based on attention weights
    updated_vectors = np.dot(attention_weights, Values)
    return updated_vectors

# Process the two radically different context tracks
sentence_agriculture = ["banana", "yellow", "fruit"]
updated_agri = run_contextual_attention(sentence_agriculture)

sentence_technology  = ["banana", "nano", "tech"]
updated_tech = run_contextual_attention(sentence_technology)

# Extract the modified coordinates for 'banana'
banana_static = E[vocab["banana"]]
banana_agri   = updated_agri[0]
banana_tech   = updated_tech[0]

# =====================================================================
# STEP 3: INTERACTIVE 3D VISUALIZATION
# =====================================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot our standard static vocabulary context points
for word, idx in vocab.items():
    if word != "banana":
        ax.scatter(E[idx][0], E[idx][1], E[idx][2], color='gray', s=100, alpha=0.6)
        ax.text(E[idx][0]+0.02, E[idx][1]+0.02, E[idx][2]+0.02, word, color='dimgray', fontsize=12)

# Plot the transformation trajectory of the word 'banana'
ax.scatter(banana_static[0], banana_static[1], banana_static[2], color='black', s=200, marker='o', label='Static Banana (Base)')
ax.scatter(banana_agri[0], banana_agri[1], banana_agri[2], color='green', s=200, marker='^', label='Banana (Agri Context)')
ax.scatter(banana_tech[0], banana_tech[1], banana_tech[2], color='blue', s=200, marker='s', label='Banana (Tech Context)')

# Draw paths showing the geometric shift caused by attention
ax.plot([banana_static[0], banana_agri[0]], [banana_static[1], banana_agri[1]], [banana_static[2], banana_agri[2]], color='green', linestyle='--', linewidth=2)
ax.plot([banana_static[0], banana_tech[0]], [banana_static[1], banana_tech[1]], [banana_static[2], banana_tech[2]], color='blue', linestyle='--', linewidth=2)

# Graph configurations
ax.set_title("The Trajectory of Meaning: Attention Vectors in 3D Space", fontsize=16, pad=20)
ax.set_xlabel("Food / Biology Axis", fontsize=11)
ax.set_ylabel("Physics / Nano Axis", fontsize=11)
ax.set_zlabel("Yellow Color Axis", fontsize=11)
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.1, 1.2)
ax.set_zlim(-0.1, 1.2)
ax.legend(loc='upper left', fontsize=11)
plt.show()

# Print mathematical validation straight to console
print("\n=== GEOMETRIC VERIFICATION ===")
print(f"Initial Baseline Banana:       {banana_static}")
print(f"Contextual Agriculture Banana: {banana_agri.round(3)}")
print(f"Contextual Technology Banana:  {banana_tech.round(3)}")
print("-" * 40)
print(f"Similarity to 'nano' in AGRI context: {cosine_similarity(banana_agri, E[vocab['nano']]):.4f}")
print(f"Similarity to 'nano' in TECH context: {cosine_similarity(banana_tech, E[vocab['nano']]):.4f}\n")