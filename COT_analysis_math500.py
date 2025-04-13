import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style="whitegrid")


def analyze_dataset(filepath, dataset_name):
    print("\n" + "=" * 60)
    print(f"📁 Starting analysis for: {dataset_name}")
    print("=" * 60)

    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(filepath)
    df["correct"] = df["correct"].astype(bool)

    # Basic statistics
    print("📊 Calculating statistics...")
    total = len(df)
    num_correct = df["correct"].sum()
    accuracy = num_correct / total
    # num_invalid = df["predicted_answer"].apply(lambda x: str(x).strip().upper() == "INVALID").sum()

    print(f"✅ Total Samples: {total}")
    print(f"✅ Accuracy: {accuracy:.2%}")
    # print(f"❌ INVALID Predictions: {num_invalid}")

    # Figure 1: Accuracy pie chart
    print("📈 Plot 1: Accuracy distribution pie chart")
    correct_count = df["correct"].sum()
    incorrect_count = len(df) - correct_count

    plt.figure(figsize=(5, 5))
    plt.pie(
        [correct_count, incorrect_count],
        labels=["Correct", "Incorrect"],
        autopct="%.1f%%",
        colors=["#90ee90", "#ff9999"],
        startangle=90
    )
    plt.title(f"{dataset_name} - Accuracy Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # Figure 2: Response length boxplot
    print("📈 Plot 2: Response length vs correctness")
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="correct", y="response_length", data=df, palette="Set2")
    plt.title(f"{dataset_name} - Response Length by Correctness")
    plt.xlabel("Correct")
    plt.ylabel("Response Length (characters)")
    plt.tight_layout()
    plt.show()

    # Figure 3: Prompt vs Completion Tokens
    print("📈 Plot 3: Prompt tokens vs completion tokens")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x="prompt_tokens",
        y="completion_tokens",
        hue="correct",
        data=df,
        palette={True: "green", False: "red"},
        alpha=0.6
    )
    plt.title(f"{dataset_name} - Prompt vs Completion Tokens")
    plt.xlabel("Prompt Tokens")
    plt.ylabel("Completion Tokens")
    plt.tight_layout()
    plt.show()

    # Figure 4: Total token histogram
    print("📈 Plot 4: Total token usage distribution")
    plt.figure(figsize=(7, 4))
    sns.histplot(df["total_tokens"], bins=20, kde=True, color="skyblue")
    plt.title(f"{dataset_name} - Total Token Usage Distribution")
    plt.xlabel("Total Tokens")
    plt.tight_layout()
    plt.show()

    # Figure 5: Confidence violin plot
    print("📈 Plot 5: Confidence distribution by correctness")
    plt.figure(figsize=(6, 4))
    sns.violinplot(
        x="correct",
        y="confidence",
        data=df.astype({"correct": str}),
        palette={"True": "#90ee90", "False": "#ff9999"},
        inner="quartile"
    )
    plt.title(f"{dataset_name} - Confidence by Correctness")
    plt.xlabel("Correct")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()

    print(f"✅ Finished analysis for {dataset_name}")
    print("=" * 60 + "\n")


# === Run for both datasets ===
analyze_dataset("results/COT_math500_qwen.csv", "MATH-500 (Qwen)")