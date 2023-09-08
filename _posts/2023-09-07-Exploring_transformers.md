---
layout: post
title: Attention is All you Need
description: Exploring transformers and AI’s secret to understand language and More.
date: 2023-09-07
tags: machine learning algorithms, interview preperation, job hunting, publication
comments_id: 1
---
# **Attention is All you Need - Exploring transformers and AI’s secret to understand language**

There has been a rapid growth in the domain of Generative AI and the large language models. These complex systems, like OpenAI’s GPT, can understand and generate text with similar capabilities to humans. This transformation is reshaping the very essence of AI technology's potential.

Over the years, the field of language modeling has seen significant evolution and advancement. 

[<img align="center" src="/assets/LLM_timeline.PNG" width="600"/>](/assets/LLM_timeline.png)


**Before 1990s**: The early days of language modeling were marked by rule-based systems and handcrafted grammars. These models were limited in their ability to understand language nuances and lacked the capabilities to handle real-world language data effectively.

**1990 - 2000**: The years between 1990s to 2000s introduced statistical approaches to language modeling. The advent of N-grams and Hidden Markov Models (HMMs) enabled computers to probabilistically predict the next word in a sequence based on historical data.

**2000 - 2010**: Some models even combined rule-based and statistical approaches to improve accuracy. 

**2010-2018**: The most significant leap in language modeling came with the rise of neural networks, particularly deep learning techniques. Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and eventually, Transformers revolutionized the field. Transformers, introduced in 2017, have become the foundation for state-of-the-art language models.

**2018 onwards**: The recent trend in language modeling is the use of pretrained models. Large-scale pretrained models, like BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pretrained Transformer), have achieved remarkable results. These models are pretrained on vast amounts of text data and can be fine-tuned for specific NLP tasks, making them versatile and highly effective.

The latest phase involves models that can process multiple modalities of data, such as text and images. This expansion into multimodal AI has opened up new possibilities for understanding and generating content that combines text and visual elements.

## Difference between RNNs, LSTMs and Transformers:  

RNNs and LSTMs introduced the concept of sequential data processing, making them suitable for tasks like machine translation and text generation. However, they struggled with long-range dependencies in language.

Transformers, with their attention mechanisms, addressed the limitation of handling long-range dependencies. They could efficiently capture contextual information from across the input sequence, leading to significant improvements in various natural language processing (NLP) tasks.

[<img align="center" src="/assets/diff_lstms_trx.PNG" width="600"/>](/assets/diff_lstms_trx.PNG)

Add difference table. 

## Transformer Architecture

[<img align="center" src="/assets/transformer_architecture.PNG" width="600"/>](/assets/transformer_architecture.png)

Now lets dive into the architecture of a transformer. 
Imagine you're reading a book, and you want to understand the meaning of a sentence. You don't just look at one word; you consider how each word in the sentence relates to the others. That's what the Transformer does, but it does it incredibly fast and with a lot of sentences all at once.

Here are the key parts:

**Attention**: Think of it as paying more attention to some words in a sentence than others. For example, in the sentence "The cat sat on the mat," the Transformer knows that "cat" and "mat" are related because they're both about where the cat is.

**Multi-Head Attentio**n: This means it can focus on different things in different sentences simultaneously. It's like reading many books at once!

**Positional Info**: When we read, the order of words matters. "Cat sat" is different from "Sat cat." Transformers understand this too.

**Encoder and Decoder**: It's like having a smart reader (encoder) and a writer (decoder). The smart reader understands the sentences, and the writer can create new ones.

**Layering**: Transformers don't just read once; they read many times, getting better with each pass. It's like thinking deeper and deeper about a story.

**Using in Many Tasks**: People use Transformers to understand languages, translate languages, summarize articles, and even help computers "see" images better!

## BERT vs Transformer

BERT (Bidirectional Encoder Representations from Transformers) is a specific model that utilizes the Transformer architecture. The main difference between BERT and the original Transformer is in how they are trained and used.

**Training Approach:**

BERT: It's pre-trained using a masked language modeling objective. BERT learns to predict missing words in sentences, which helps it understand context bidirectionally (both left and right context).
Transformer: The original Transformer doesn't have this pre-training phase. It's typically trained for specific tasks from scratch.
Bidirectionality:

BERT: BERT can look at both the left and right context of a word to understand its meaning within a sentence.
Transformer: The original Transformer is unidirectional, meaning it processes words sequentially from left to right or right to left.
Usage:

BERT: It's often used as a pre-trained language model. Fine-tuning is performed on top of the pre-trained BERT model for specific tasks like text classification, question answering, and more.
Transformer: The original Transformer can also be used for various NLP tasks, but it usually requires more task-specific training.

## GPT vs Transformer

GPT (Generative Pre-trained Transformer) is another model that builds upon the Transformer architecture. Here are the key differences between GPT and the original Transformer:

Training Objective:

GPT: It's trained for language modeling. GPT learns to predict the next word in a sentence, given the context of the previous words. This allows it to generate coherent and contextually relevant text.
Transformer: The original Transformer doesn't have a specific pre-training objective like language modeling. It's often used as a building block for various NLP tasks.
Use Cases:

GPT: GPT models are primarily used for text generation tasks, such as chatbots, content generation, and language translation. They excel at generating human-like text.
Transformer: The original Transformer is a more general architecture that can be adapted to various NLP tasks, including text classification, machine translation, and more.
Bidirectionality:

GPT: GPT models are unidirectional. They process words sequentially from left to right.
Transformer: The original Transformer can be bidirectional or unidirectional depending on how it's configured.
Fine-tuning:

GPT: GPT models can be fine-tuned for specific text generation tasks with relatively little task-specific data.
Transformer: Fine-tuning the original Transformer for specific tasks may require more task-specific data and customization.

Transformers showcased groundbreaking advancement in the field of artificial intelligence, particularly in natural language processing.As the field of AI advances, we can expect more sophisticated, efficient, and versatile Large Language Models, each pushing the boundaries of what artificial intelligence can achieve.