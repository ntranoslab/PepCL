# PepCL

[![Tests](https://github.com/ntranoslab/PepCL/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ntranoslab/PepCL/actions/workflows/tests.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntranoslab/PepCL/blob/main/notebooks/01_pepcl_quickstart.ipynb)

PepCL is a replay-based continual learning framework for adapting peptide-MHC class I prediction models to new datasets while retaining performance on the pretrained MS domain.

PepCL builds on [MHCPrime](https://github.com/ntranoslab/MHCPrime), which provides the base model, tokenizer, preprocessing utilities, and inference functions used throughout the quickstart workflow.

<p align="center">
  <img src="assets/pepcl_schema.png" alt="PepCL schema" width="900">
</p>