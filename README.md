# language-modeling-cometh
This repo stores the work I participated in the URP, COMETH, at RPI.

## preprocess corpus
we preprocess the corpus for fine-tuning

```python
defineCorpus(publishers, passages, documents, output)
```

## Fine-tune bert model
we use 10%, 10% and 80% propotions of the corpus for Test, Validation and Training, respectively to fine-tune the pretrained bert model
```python
fineTuneEmbedding(publishers, output)
```

## Combine with Flair Transformer
we combine the fine-tuned bert models with Flair Transformer
```python
combineModelFlair(publishers, documents, ignores, output)
```

## Embedding
we use Flair embedding for its Transformer
```python
saveEmbedding(embeddingVectors, output)
```
