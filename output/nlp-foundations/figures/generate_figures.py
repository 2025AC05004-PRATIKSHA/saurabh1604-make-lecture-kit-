import sys, os
sys.path.insert(0, os.path.abspath('scripts'))
import numpy as np
import matplotlib.pyplot as plt
import figstyle

outdir = os.path.dirname(os.path.abspath(__file__))
figstyle.use_house_style()

# 1. Linguistic levels
steps = [
    "Pragmatics & Discourse\n(Social intent & multi-sentence flow)",
    "Semantics\n(Literal word & compositional meaning)",
    "Syntax\n(Grammar trees & phrase structures)",
    "Morphology\n(Word formation, morphemes & stems)",
    "Phonetics & Orthography\n(Speech sounds, letters & spelling)"
]
figstyle.flow(steps, "The Six Levels of Linguistic Analysis", direction="tb", out=os.path.join(outdir, "fig_linguistic_levels.png"))

# 2. Confusion matrix heatmap
cm_data = [[35, 10], [5, 50]]
figstyle.heatmap(cm_data, "Confusion Matrix (SMS Spam Classification)",
                 row_labels=["Actual Spam (+)", "Actual Ham (-)"],
                 col_labels=["Pred Spam (+)", "Pred Ham (-)"],
                 xlabel="Predicted Class", ylabel="Actual Class",
                 out=os.path.join(outdir, "fig_confusion_matrix.png"))

# 3. Text Preprocessing Flow
prep_steps = [
    "Raw Text\n'Cats 100%'",
    "Cleaned Text\n'cats 100'",
    "Tokens\n['cats', '100']",
    "Stopword Filter\nRemove 'the', 'is'",
    "Normalized Terms\nStem / Lemmatize"
]
figstyle.flow(prep_steps, "Text Preprocessing Pipeline", direction="lr", out=os.path.join(outdir, "fig_preprocessing_pipeline.png"))

# 4. Cosine vectors (cherry, digital, information)
norm_cherry = (0.95, 0.05, "cherry (pie-focused)", figstyle.PALETTE["high"])
norm_digital = (0.05, 0.88, "digital (data/comp)", figstyle.PALETTE["blue"])
norm_info = (0.04, 0.96, "information (data/comp)", figstyle.PALETTE["accent"])
figstyle.vectors2d([norm_cherry, norm_digital, norm_info],
                   "Word Vectors: Angle Reveals Meaning",
                   xlabel="Context: 'pie' dimension", ylabel="Context: 'computer / data' dimension",
                   xlim=(-0.1, 1.1), ylim=(-0.1, 1.2),
                   out=os.path.join(outdir, "fig_cosine_vectors.png"))

# 5. TF-IDF Weighting curves
x_counts = np.linspace(0, 100, 200)
tf_raw = x_counts / 100.0
tf_log = np.log10(x_counts + 1) / np.log10(101)
figstyle.curve(x_counts,
               {"Raw Frequency (linear)": tf_raw, "Sublinear TF: log10(count + 1)": tf_log},
               "Sublinear Scaling Prevents Frequent Words from Dominating",
               xlabel="Term Count in Document", ylabel="Normalized Scaled Weight",
               out=os.path.join(outdir, "fig_tfidf_weighting.png"))

# 6. Skip-gram vs CBOW architectures
cbow_steps = [
    "Context Words\n[c1, c2, c3, c4]",
    "Average Projection / Hidden",
    "Predict Center\nw_target"
]
figstyle.flow(cbow_steps, "CBOW: Context Predicts Center Word", direction="lr", out=os.path.join(outdir, "fig_cbow_architecture.png"))

sg_steps = [
    "Center Target\nw_target",
    "Target Embedding\nw in W",
    "Predict Contexts\n[c1, c2, c3, c4]"
]
figstyle.flow(sg_steps, "Skip-gram: Center Word Predicts Context Window", direction="lr", out=os.path.join(outdir, "fig_skipgram_architecture.png"))

# 7. SGNS Gradient update
vectors_sgns = [
    (1.2, 0.2, "target: Ned (w)", figstyle.PALETTE["ink"]),
    (1.0, 0.9, "pos context: Stark (c_pos)", figstyle.PALETTE["blue"]),
    (-0.8, -0.6, "neg: pimples (c_neg1)", figstyle.PALETTE["red"]),
    (-0.5, 0.8, "neg: zebra (c_neg2)", figstyle.PALETTE["red"]),
    (0.1, -1.0, "neg: idiot (c_neg3)", figstyle.PALETTE["red"])
]
figstyle.vectors2d(vectors_sgns,
                   "SGNS Learning: Pull Positives Closer, Push Negatives Away",
                   xlabel="Latent Semantic Axis 1", ylabel="Latent Semantic Axis 2",
                   xlim=(-1.2, 1.6), ylim=(-1.3, 1.3),
                   out=os.path.join(outdir, "fig_sgns_gradient_step.png"))

# 8. Analogy parallelogram
analogy_vecs = [
    (0.2, 0.8, "King", figstyle.PALETTE["blue"]),
    (0.1, 0.2, "Man", figstyle.PALETTE["muted"]),
    (0.8, 0.2, "Woman", figstyle.PALETTE["muted"]),
    (0.9, 0.8, "Queen (King - Man + Woman)", figstyle.PALETTE["high"])
]
figstyle.vectors2d(analogy_vecs,
                   "Vector Parallelogram Model of Analogies",
                   xlabel="Semantic Dimension 1 (Gender)", ylabel="Semantic Dimension 2 (Royalty)",
                   xlim=(-0.1, 1.2), ylim=(-0.1, 1.1),
                   out=os.path.join(outdir, "fig_analogies_parallelogram.png"))

print("Figures generated successfully!")
