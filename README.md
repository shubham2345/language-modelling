# Language Modeling with Transformers

This repository contains a project focused on implementing a language modeling task using a Transformer-based architecture. The model leverages attention mechanisms, SentencePiece tokenization, and FastText embeddings to predict the next token in a sequence, ultimately evaluating its performance through perplexity.

## Overview

Language modeling is a fundamental task in natural language processing, aiming to predict the next token in a sequence based on the previous tokens. This project explores various architectures and tokenization strategies to improve the accuracy and efficiency of the model. Key highlights include:

- Transformer-based architecture
- Attention mechanisms
- SentencePiece tokenization
- FastText embeddings
- Perplexity evaluation

## How It Works

Given a sequence of tokens, the model predicts the probability distribution of the next token. The predictions are evaluated using perplexity, which measures how well a probabilistic model predicts a sample.

Example sequence:

```
<s> NLP 243 is the best </s>
```

Predictions:

- p(NLP | <s>)
- p(243 | <s>, NLP)
- p(is | <s>, NLP, 243)
- p(the | <s>, NLP, 243, is)
- p(best | <s>, NLP, 243, is, the)

## Requirements

- Python 3.8+
- Install the required libraries using:
  ```bash
  pip install -r requirements.txt
  ```

## Running the Code

To train the model and generate predictions, run the following command:

```bash
python run.py submission.csv
```

Ensure that your input data is correctly formatted and placed in the appropriate directory.

## Model Evaluation

The performance of the language model is evaluated using perplexity:

```
exp(-1/T * sum(log p(t_i | t_<i)))
```

Where:

- **T**: Total number of tokens in the sentence
- **p(t_i | t_<i)**: Predicted probability of token t_i given the previous tokens

## Submission Format

Your final predictions should be saved in a CSV file with the following format:

```
ID,ppl
2,2.134
5,5.230
6,1.120
```

- **ID**: Sentence ID
- **ppl**: Perplexity value

## Techniques Explored

The project explores various techniques to improve the model's performance:

- **Recurrent Neural Networks (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Units (GRU)**
- **Transformers**
- **Attention mechanisms**
- **Different tokenization strategies**
- **Pretrained embeddings (Word2Vec, GloVe)**

## Limitations
External libraries such as HuggingFace Transformers or Keras were not used for model implementation or training. The focus was on building core components from scratch.

