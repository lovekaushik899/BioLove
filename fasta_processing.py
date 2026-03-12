#!/usr/bin/env python3

import argparse
import os
import math
import itertools
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tqdm import tqdm


########################################################
# FASTA PARSER
########################################################

def read_fasta(file):

    header = None
    seq = []

    with open(file) as f:

        for line in f:

            line = line.strip()

            if line.startswith(">"):

                if header:
                    yield header, "".join(seq)

                header = line[1:]
                seq = []

            else:
                seq.append(line.upper())

        if header:
            yield header, "".join(seq)


########################################################
# FEATURE FUNCTIONS
########################################################

def seq_length(seq):
    return len(seq)


def gc_content(seq):

    g = seq.count("G")
    c = seq.count("C")

    return (g + c) / len(seq)


def at_content(seq):

    a = seq.count("A")
    t = seq.count("T")

    return (a + t) / len(seq)


def gc_skew(seq):

    g = seq.count("G")
    c = seq.count("C")

    if g + c == 0:
        return 0

    return (g - c) / (g + c)


########################################################
# MONONUCLEOTIDE COMPOSITION
########################################################

def mono_composition(seq):

    counts = Counter(seq)
    length = len(seq)

    return {
        "A_freq": counts.get("A", 0) / length,
        "C_freq": counts.get("C", 0) / length,
        "G_freq": counts.get("G", 0) / length,
        "T_freq": counts.get("T", 0) / length
    }


########################################################
# GENERALIZED KMER COMPOSITION
########################################################

def kmer_composition(seq, k):

    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)

    alphabet = ["A", "C", "G", "T"]
    all_kmers = ["".join(p) for p in itertools.product(alphabet, repeat=k)]

    features = {}

    for kmer in all_kmers:
        features[f"{k}mer_{kmer}"] = counts.get(kmer, 0) / len(kmers)

    return features


########################################################
# K-SPACED NUCLEOTIDE PAIRS
########################################################

def k_spaced_pairs(seq, k=1):

    counts = Counter()

    for i in range(len(seq)-k-1):

        pair = seq[i] + seq[i+k+1]
        counts[pair] += 1

    total = sum(counts.values())

    features = {}

    for a in "ACGT":
        for b in "ACGT":

            pair = a + b
            features[f"k{k}_pair_{pair}"] = counts.get(pair, 0) / total

    return features


########################################################
# ACCUMULATED NUCLEOTIDE FREQUENCY
########################################################

def accumulated_nucleotide_frequency(seq):

    L = len(seq)

    segments = [
        seq[:int(L*0.25)],
        seq[:int(L*0.50)],
        seq[:int(L*0.75)]
    ]

    features = {}

    for i, segment in enumerate(segments):

        counts = Counter(segment)

        for base in "ACGT":
            features[f"ANF_{i+1}_{base}"] = counts.get(base, 0) / len(segment)

    return features


########################################################
# SHANNON ENTROPY
########################################################

def shannon_entropy(seq):

    counts = Counter(seq)
    length = len(seq)

    entropy = 0

    for base in "ACGT":

        p = counts.get(base, 0) / length

        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


########################################################
# Z CURVE
########################################################

def z_curve(seq):

    a = seq.count("A")
    c = seq.count("C")
    g = seq.count("G")
    t = seq.count("T")

    x = (a + g) - (c + t)
    y = (a + c) - (g + t)
    z = (a + t) - (c + g)

    return {
        "Z_curve_x": x,
        "Z_curve_y": y,
        "Z_curve_z": z
    }


########################################################
# FEATURE EXTRACTION
########################################################

def extract_features(record):

    header, seq, label = record

    features = {}

    features["length"] = seq_length(seq)

    features["GC_content"] = gc_content(seq)
    features["AT_content"] = at_content(seq)
    features["GC_skew"] = gc_skew(seq)

    features.update(mono_composition(seq))

    features.update(kmer_composition(seq, 2))
    features.update(kmer_composition(seq, 3))

    features.update(k_spaced_pairs(seq, 1))

    features.update(accumulated_nucleotide_frequency(seq))

    features["entropy"] = shannon_entropy(seq)

    features.update(z_curve(seq))

    features["FASTA_Header"] = header
    features["Label"] = label

    return features


########################################################
# LOAD DATA
########################################################

def load_dataset(pos_file, neg_file):

    dataset = []

    for h, s in read_fasta(pos_file):
        dataset.append((h, s, 1))

    for h, s in read_fasta(neg_file):
        dataset.append((h, s, 0))

    return dataset


########################################################
# MODEL EVALUATION
########################################################

def evaluate_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return acc


########################################################
# INCREMENTAL FEATURE SELECTION
########################################################

def run_ifs(X, y, headers, labels, outdir):

    print("\nRunning Incremental Feature Selection\n")

    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier()
    }

    rf = RandomForestClassifier()
    rf.fit(X, y)

    importances = rf.feature_importances_

    ranked = list(zip(X.columns, importances))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    ranked = [x[0] for x in ranked]

    subset_sizes = [5, 10, 20, 50, 100]

    results = []

    for model_name, model in models.items():

        for size in subset_sizes:

            selected = ranked[:size]

            X_sub = X[selected]

            acc = evaluate_model(model, X_sub, y)

            results.append(["IFS", model_name, size, acc])

            df_out = pd.concat([headers, labels, X_sub], axis=1)

            df_out.to_csv(
                os.path.join(outdir, f"IFS_{model_name}_top{size}.csv"),
                index=False
            )

            print(f"IFS {model_name} Top{size} Accuracy {acc*100:.2f}%")

    return results


########################################################
# RECURSIVE FEATURE ELIMINATION
########################################################

def run_rfe(X, y, headers, labels, outdir):

    print("\nRunning Recursive Feature Elimination\n")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(kernel="linear"),
        "GradientBoosting": GradientBoostingClassifier()
    }

    subset_sizes = [5, 10, 20, 50, 100]

    results = []

    for model_name, model in models.items():

        for size in subset_sizes:

            rfe = RFE(model, n_features_to_select=size)

            rfe.fit(X, y)

            selected = X.columns[rfe.support_]

            acc = evaluate_model(model, X[selected], y)

            results.append(["RFE", model_name, size, acc])

            df_out = pd.concat([headers, labels, X[selected]], axis=1)

            df_out.to_csv(
                os.path.join(outdir, f"RFE_{model_name}_top{size}.csv"),
                index=False
            )

            print(f"RFE {model_name} Top{size} Accuracy {acc*100:.2f}%")

    return results


########################################################
# MAIN PIPELINE
########################################################

def run_pipeline(pos, neg, outdir, cores):

    os.makedirs(outdir, exist_ok=True)

    print("\n==============================")
    print("BioLove FASTA Processing Pipeline")
    print("==============================\n")

    dataset = load_dataset(pos, neg)

    print("Total sequences:", len(dataset))

    print("\nExtracting features\n")

    with Pool(cores) as p:

        features = list(
            tqdm(
                p.imap(extract_features, dataset),
                total=len(dataset)
            )
        )

    df = pd.DataFrame(features)

    df.to_csv(os.path.join(outdir, "consolidated_features.csv"), index=False)

    headers = df[["FASTA_Header"]]
    labels = df[["Label"]]

    X = df.drop(columns=["FASTA_Header", "Label"])
    y = df["Label"]

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    ifs_results = run_ifs(X_scaled, y, headers, labels, outdir)
    rfe_results = run_rfe(X_scaled, y, headers, labels, outdir)

    summary = pd.DataFrame(
        ifs_results + rfe_results,
        columns=["Method", "Model", "Features", "Accuracy"]
    )

    summary.to_csv(
        os.path.join(outdir, "performance_summary.csv"),
        index=False
    )

    print("\nPipeline completed\n")


########################################################
# CLI ENTRY POINT
########################################################

def main():

    parser = argparse.ArgumentParser(
        description="BioLove: FASTA Feature Extraction and Feature Selection Pipeline"
    )

    parser.add_argument("--pos", required=True, help="Positive FASTA file")
    parser.add_argument("--neg", required=True, help="Negative FASTA file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--cores", type=int, default=1, help="CPU cores")

    args = parser.parse_args()

    run_pipeline(args.pos, args.neg, args.out, args.cores)


if __name__ == "__main__":
    main()
