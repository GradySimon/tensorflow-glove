# tf-glove

##  What is this?
This is an implementation of [GloVe](http://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation), a model for learning vector representations of words. The model was originally developed by Jeffery Pennington, Richard Socher, and Christopher Manning.

This is my implementation of their model in [TensorFlow](http://www.tensorflow.org/), a "library for numerical computation using data flow graphs" by Google.

## How do I use it?

Like this:

```python
>>> import tf_glove
>>> model = tf_glove.GloVeModel(embedding_size=300, context_size=10)
>>> model.fit_to_corpus(corpus)
>>> model.train(num_epochs=100)
>>> model.embedding_for("reddit")
array([ 0.77469945,  0.06020461,
        0.37193006, -0.44537717,
        ...
        0.29987332, -0.12688215,], dtype=float32)
>>> model.generate_tsne()
```
![t-SNE visualization](https://cloud.githubusercontent.com/assets/1183957/11329891/f1682f8e-9156-11e5-8462-33ba46bfb16c.png)

For a more complete introduction, see the [Getting Started notebook](https://github.com/GradySimon/tf-glove/blob/master/Getting%20Started.ipynb).

## Credits
Naturally, most of the credit goes to Jeffery Pennington, Richard Socher, and Christopher Manning, who developed the model, published a paper about it, and released an implementation in C.

Thanks also to Jon Gauthier ([@hans](https://github.com/hans)), who wrote a Python implementation of the model and a blog post describing that implementation, which were both very useful references as well.

## References
- Pennington, J., Socher, R., & Manning, C. D. (2014). [Glove: Global vectors for word representation](http://nlp.stanford.edu/pubs/glove.pdf). Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014), 12, 1532-1543.
- [stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe) - Pennington, Socher, and Manning's C implementation of the model
- [hans/glove.py](https://github.com/hans/glove.py) - Jon Gauthier's Python implementation
- [A GloVe implementation in Python](http://www.foldl.me/2014/glove-python/) - Jon's blog post on the topic
