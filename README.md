# Contrastive Activation Addition

This repository contains a notebook that reproduces the workflow for extracting steering vectors from input, as defined in the Contrastive Activation Addition (CAA) paper.

## Overview

Contrastive Activation Addition (CAA) is a technique for extracting steering vectors that modify the behavior of language models. This project demonstrates the process using Llama-2 chat models.

## Features

- **Model Loading**: Supports Llama-2 models with 8-bit quantization to optimize VRAM usage.
- **Dataset Handling**: Utilizes sycophancy and corrigibility datasets formatted in the style of Anthropic's Model-Written Evals.
- **Evaluation**: Provides utilities to evaluate model performance with and without steering vectors.
- **Steering Vectors**: Implements methods to train and apply steering vectors to adjust model behavior.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install steering-vectors torch accelerate bitsandbytes ipywidgets python-dotenv
