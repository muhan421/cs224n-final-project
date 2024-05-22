# Multi-Task Inference Time Interventions of LLMs

This repository contains a notebook that demonstrates the process of extracting steering vectors from input for multi-task inference time interventions of large language models (LLMs).

## Overview

Multi-Task Inference Time Interventions involve extracting and applying steering vectors to modify the behavior of language models across different tasks. This project showcases the process using Llama-2 chat models.

## Features

- **Model Loading**: Supports Llama-2 models with 8-bit quantization to optimize VRAM usage.
- **Dataset Handling**: Utilizes sycophancy and corrigibility datasets formatted in the style of Anthropic's Model-Written Evals.
- **Evaluation**: Provides utilities to evaluate model performance with and without steering vectors.
- **Steering Vectors**: Implements methods to train and apply steering vectors to adjust model behavior for different tasks.

# Installation

To set up the environment, install the required dependencies:

```bash
pip install steering-vectors torch accelerate bitsandbytes ipywidgets python-dotenv
```

# Usage
Set up the Model: Load the Llama-2 chat models using Huggingface Transformers.
Prepare the Dataset: Download and preprocess the datasets for sycophancy and corrigibility.
Evaluate the Model: Run evaluations to assess the model's performance on the provided datasets.
Train Steering Vectors: Train steering vectors to modify the model's behavior based on the dataset.
Apply Steering Vectors: Apply the trained steering vectors and observe the changes in model performance.

# Colab
You can run this notebook in Google Colab. Note that the standard T4 GPU available with the free tier supports the Llama-2-7b model but not the Llama-2-13b model.

## Acknowledgements
This project builds upon the work of Rimsky et al. in the Contrastive Activation Addition (CAA) paper. The official codebase and further details can be found here.
