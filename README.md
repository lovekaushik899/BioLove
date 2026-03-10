# BioLove

BioLove is a command-line bioinformatics toolkit for automated FASTA sequence feature extraction and machine-learning driven feature selection.

The tool transforms biological sequences into structured feature matrices and performs independent feature selection using Incremental Feature Selection (IFS) and Recursive Feature Elimination (RFE) across multiple machine learning models.

---

## Workflow

![BioLove Workflow](BioLove.png)

*Workflow illustrating automated FASTA sequence feature extraction, dataset construction, and independent feature selection using Incremental Feature Selection (IFS) and Recursive Feature Elimination (RFE), followed by multi-model evaluation.*

---

## Key Features

- FASTA sequence parsing
- Large-scale feature extraction
- Nucleotide composition descriptors
- Dinucleotide and trinucleotide frequencies
- GC content and GC skew
- Shannon entropy
- Z-curve representation
- Data analysis and feature normalization
- Independent feature selection pipelines:
  - Incremental Feature Selection (IFS)
  - Recursive Feature Elimination (RFE)
- Multiple machine learning models
- Multi-core parallel processing
- Structured CSV outputs ready for downstream modelling

---

## Installation

Install directly from PyPI:

```bash
pip install biolove
