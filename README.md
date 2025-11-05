# English PoS Taggers Benchmark

This repository presents a benchmark of leading open-source English part-of-speech (PoS) taggers, evaluating their accuracy using the Universal Dependencies English Web Treebank (UD-EWT) dataset.

---

## TL;DR

The table below summarizes the benchmark results across key metrics:

| Model              | Size      | Accuracy | Precision | Recall | F1     | ADJ+NOUN F1 | Throughput* (wps) |
|--------------------|-----------|----------|-----------|--------|--------|-------------|--------------|
| UDPipe             | ~17 MB    | 0.9246   | 0.9126    | 0.9246 | 0.9181 | 0.8989      | 11542        |
| Stanford CoreNLP   | ~50 MB    | **0.9474** | **0.9361** | **0.9474** | **0.9412** | **0.9420** | 3804         |
| TreeTagger         | ~3 MB     | 0.7566   | 0.8658    | 0.7566 | 0.7685 | 0.8683      | **133456**   |
| spaCy              | ~12 MB    | 0.9119   | 0.9049    | 0.9119 | 0.9061 | 0.9088      | 11237        |
| NLTK               | **~1 MB** | 0.8384   | 0.8276    | 0.8384 | 0.8270 | 0.7932      | 117256       |
| Flair              | ~30 MB    | 0.9217   | 0.9154    | 0.9217 | 0.9159 | 0.9239      | 1613         |

**Note.** Throughput (words per second) was measured on an Apple M4 Max with 36 GB RAM. Values are for relative 
comparison only and will vary by hardware, dataset, and configuration.

Among all evaluated models, [Stanford Stanza](https://stanfordnlp.github.io/stanza/) achieves the **highest overall accuracy and tagging consistency**, making it the recommended baseline for most research and production use cases.  
However, its larger model size results in **higher latency** compared to lighter alternatives.  

If **speed or memory efficiency** is a priority, **spaCy** or **UDPipe** provides a strong balance between performance and accuracy.  
For lightweight or Python-native projects where maximum throughput is required, **NLTK** remains a practical fallback despite its lower tag quality.


Evaluations were performed using the **Universal Dependencies English Web Treebank (UD_English-EWT)** dataset.  
Reported *Accuracy*, *Precision*, *Recall*, and *F1* metrics are computed using **micro-averaged tag-level classification**, while *ADJ+NOUN F1* measures sequence-level correctness for adjective–noun phrase prediction (see [Sequence Tagging](#sequence-tagging) for details).  

This benchmark focuses exclusively on **XPOS tag predictions**.

---

## Table of Contents
- [TLDR](#tldr)
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
  - [Tags consistency](#tags-consistency)
- [Metrics](#metrics)
- [Sequence tagging](#sequence-tagging)
- [Results](#results)
- [Future work](#future-work)
  - [Dataset](#dataset-1)
  - [UPOS results](#upos-results)
- [Licence](#licence)
- [Contributions](#contributions)

---

## Introduction

This repository provides a comprehensive and independent benchmark of open-source **Part-of-Speech (PoS) taggers for English**.  
It evaluates six widely used models across multiple metrics relevant to both research and applied NLP, including accuracy, precision, recall, and sequence-level tagging performance.

The benchmark aims to answer a practical question faced by many NLP practitioners and researchers:

**Which open-source PoS tagger offers the best trade-off between accuracy, speed, and usability for English text?**


All models were tested on the **Universal Dependencies English Web Treebank (UD_English-EWT)** dataset, a widely adopted corpus for syntactic and morphological analysis.

In addition to traditional token-level metrics, this benchmark includes **sequence-level evaluations** (e.g., adjective–noun combinations) that better reflect the needs of downstream applications such as **keyphrase extraction**, **entity recognition**, and **semantic analysis**.

Overall, this benchmark provides:
- A unified performance comparison of major English PoS taggers  
- Guidance for model selection based on latency, size, and accuracy  


---

## Dataset

All evaluations were conducted using the **Universal Dependencies English Web Treebank (UD_English-EWT)** dataset  
([GitHub repository](https://github.com/UniversalDependencies/UD_English-EWT?tab=readme-ov-file)).

As some models were trained using this dataset 
(UDPipe itself and [Stanford Stanza](https://stanfordnlp.github.io/stanza/available_models.html#available-ud-models)), 
`ud_dataset_test.csv` corpus was exclusively used for evaluation. 
Using `ud_dataset.csv` results in an increase in prediction quality for these models, 
which serves as evidence that the models were not trained on the `ud_dataset_test.csv` 
data and that *no data leakage affects the benchmark*.


The benchmark focuses on **XPOS** tags, as defined by the [Penn Treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).  
It should be noted that **the dataset itself contains automatically labeled tags**, which may include occasional inaccuracies.  
For instance, in the sentence:

> *In June and July of 1973, he accumulated 56 points, enough to meet the minimum requirement for the 1973–1974 year.*

Here the word *minimum* is labeled as a *noun*, though it functions as an *adjective* in this context.

Despite these limitations, **UD_English-EWT** remains one of the most reliable open-source corpora for PoS tagging research, 
offering consistent annotation guidelines and extensive linguistic coverage across diverse web-based English sources.


---

## Models

The benchmark includes six widely used open-source English part-of-speech taggers.  
All models were evaluated using their **default pretrained English models** and configured for **XPOS tagging** where supported.

| # | Model | Version | Description |
|---|--------|----------|-------------|
| 1 | **NLTK** (`averaged_perceptron_tagger_eng`) | 3.9.0 | A lightweight, rule- and feature-based tagger using an averaged perceptron model. |
| 2 | **Flair** (`pos-fast`) | 0.15.1 | A contextual string embedding model that leverages character-level recurrent representations. |
| 3 | **Stanza (Stanford NLP)** | 1.11.0 | A neural pipeline based on BiLSTM-CRF architecture, trained on Universal Dependencies data. |
| 4 | **UDPipe 1** (`english-ewt-ud-2.5-191206`) | 2019.12.06 | A neural network–based tagger trained on the UD English EWT dataset; results may reflect data overlap. |
| 5 | **TreeTagger (English)** | 3.2.3 | A probabilistic decision-tree-based PoS tagger designed for linguistic research; non-commercial license. |
| 6 | **spaCy** (`en_core_web_sm`) | 3.8.0 | An efficient CNN-based NLP pipeline optimized for production and real-time tagging. |

> ⚠️ **Licensing note:** TreeTagger is distributed under a **non-commercial license** and is **not included** in this repository.  
> Users wishing to reproduce its results must download it separately from its [official source](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) and comply with its original license terms.

Only models that support **XPOS tagging** were included in this benchmark.

---

## Methodology

Using open source labeled dataset all models where run sentence by sentence and predicted PoS tags where gain. 
Only XPOS (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) where evaluated.
The train and test datasets were merged to increase size. 

---

### Tags consistency

Flair, Stanza, Spacy, NLTK and UD models shares standard Penn-treebanck XPOS tags. 
However TreeTagger does not distinguish some fine-grained noun PoS tags. 
Dataset tags was not changed so some portion of accuracy for that model is lost due to inconsistent tagging.

---

## Metrics

Several metrics where calculated: 

**Global Metrics**

1. Accuracy
2. Precision
3. Recall
4. F1 score

This is by tag classification. In both micro and macro versions.
Per-sentence metrics also were calculated. These are true when all tags in a sentence predicted correctly.

You can find detailed metrics for each model in metrics.json file located in models directory. 

You can also find confusion matrix and per class metrics for each model visiting model directory. 
For practical issues additional metrics where evaluated as well.

---

## Sequence tagging 

While PoS tagging is a low-level task that is unlikely to be directly used as input for production 
systems, PoS taggers can serve as components within more complex machine learning systems, 
such as *keyphrase extraction* algorithms. PoS information can help other models identify keyphrases in text.
For instance, an **Adjective + Noun** combination has a higher likelihood of representing a keyphrase than 
a **Noun + Verb** pair, making PoS tags a relevant feature.

To address this aspect specifically, this benchmark provides fine-grained (micro-level) metrics for these tag combinations:


1. ADJ + NOUN
2. NOUN + NOUN
3. PRON + PRON
4. ADJ + PRON

You can find detailed metrics for each model in metrics.json file located in models directory.

---

## Results

The overview table with most valuable metrics is located in TL;DR section. 

**Stanford Stanza** PoS tagger demonstrates the highest overall prediction quality. 
Therefore, it can be recommended as a strong baseline.
However, as the largest model, it also has the lowest inference speed.

If the performance is critical factor, consider **SpaCy** or **UDPipe** as faster alternatives. 

The **NLTK** tagger is the smallest and fastest among the tested models. 
Thus, for applications where processing speed is more important than prediction quality,
NLTK may be a suitable choice—particularly if the project is Python-native.

Note that TreeTagger is for non-commercial use only. 

---

## Future work

There are several uncovered topics worth investigating. 

### Dataset

While the current dataset is considered an industry standard in NLP,
it primarily consists of media-domain data. Although that evaluation is valuable for this domain, 
model performance on other data such as scientific texts could differ significantly. 
 
It is also of interest to explore the use of large language models (LLMs) as PoS taggers.

Although there are relatively few studies on using LLMs for PoS tagging, works such as  
[Alhessi & Bird (2024)](https://arxiv.org/abs/2404.18286) demonstrate that LLMs can reach 98%+ accuracy on labeled datasets.
However, potential data leakage remains a concern, since publicly available datasets could be included in the training dataset. 
Nevertheless, even imperfect tags can still be used to assess the performance of different models, 
providing useful and practically relevant information.

---

### UPOS results
Currently only XPOS tags performance is evaluated. It is planned to assess it for UPOS.

---

## Licence

All original code and benchmark materials in this repository are released under the [MIT License](LICENSE), 
and may be freely used, modified, and distributed for both commercial and non-commercial purposes.

However, please note:

- **TreeTagger** (by Helmut Schmid, University of Stuttgart) is **not included** in this repository and is licensed separately 
  for **non-commercial use only**.  
- Any results or references to TreeTagger in this benchmark were generated using the publicly available 
  non-commercial version.  
- If you wish to reproduce the TreeTagger results, you must download it from its [official website](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) 
  and comply with its original license terms.

---

## Citations

If you use this benchmark, please cite this repository and the original tools/datasets below.

**This repository**

Ivan Smirnov. (2025). *English PoS Taggers Benchmark* (Version 0.1.0). GitHub. https://github.com/ivLoneas/pos-taggers-benchmark

**NLTK**

Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O’Reilly Media.

**Flair**

Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S., & Vollgraf, R. (2019). Flair: An easy-to-use framework for state-of-the-art NLP. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)*, 54–59.

**Stanza (Stanford NLP)**

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, 101–108.

**spaCy**

Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing*. Software documentation / preprint.

**UDPipe**

Straka, M., & Straková, J. (2017). Tokenizing, POS tagging, lemmatizing and parsing UD 2.0 with UDPipe. *Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies*, 88–99.

**TreeTagger**

Schmid, H. (1994). Probabilistic part-of-speech tagging using decision trees. *Proceedings of the International Conference on New Methods in Language Processing*.

**UD_English-EWT Dataset**

Silveira, N., Dozat, T., de Marneffe, M.-C., Bowman, S., Connor, M., Bauer, J., & Manning, C. D. (2014). A gold standard dependency corpus for English: The English Web Treebank. *Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14)*.

**Related / Future Work**

Alhessi, A., & Bird, S. (2024). *Evaluating Large Language Models on Part-of-Speech Tagging.* arXiv preprint [arXiv:2404.18286](https://arxiv.org/abs/2404.18286).

---

## Contributions

These evaluations are open source, so everyone is welcome to use them for any purpose.

That said, I would highly appreciate any mentioning and starring that GitHub repo — it helps a lot!

If you have any questions or suggestions, feel free to reach me at:

`ivan.smirnov.wrk@gmail.com`

— Ivan

---
