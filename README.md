# Multi-Task Alignment of LLMs Using Steering Vectors

This repository contains a collection of scripts and notebooks that demonstrate the process of extracting and applying steering vectors for multi-task inference time interventions of large language models (LLMs).

## Overview

Multi-Task Inference Time Interventions involve extracting and applying steering vectors to modify the behavior of language models across different tasks. This project showcases the process using LLaMA-2 chat models.

## Features

- **Model Loading**: Supports LLaMA-2 models with 8-bit quantization to optimize VRAM usage.
- **Dataset Handling**: Utilizes sycophancy, corrigibility, and truthfulQA datasets formatted in the style of Anthropic's Model-Written Evals.
- **Evaluation**: Provides utilities to evaluate model performance with and without steering vectors on sycophancy, corrigibility, and truthfulness tasks.
- **Steering Vectors**: Implements methods to train and apply steering vectors to adjust model behavior for different tasks. 
- **Open-Ended Evaluation**: Uses GPT-4 via the OpenAI API for evaluating open-ended responses.
- **General Capability Evaluation**: Assesses model performance on the MMLU dataset to ensure general capabilities are preserved.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install steering-vectors torch accelerate bitsandbytes ipywidgets python-dotenv
```

## Acknowledgements
This project builds upon the work of Rimsky et al. in the Contrastive Activation Addition (CAA) paper. The official codebase and further details can be found here.
