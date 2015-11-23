# GloVe: Global Vectors for Word Representation
### in TensorFlow

##  What is this?
This is an implementation of [GloVe](http://nlp.stanford.edu/projects/glove/), a model for learning word representations by Jeffery Pennington, Richard Socher, and Christopher Manning (none of whom are me). The model is built in [TensorFlow](http://www.tensorflow.org/), a "library for numerical computation using data flow graphs" by Google.

## Credits
Naturally, most of the credit goes to Jeffery Pennington, Richard Socher, and Christopher Manning, who developed the model, published a paper about it (see references), and released an implementation of the model in C.

Thanks also to Jon Gauthier ([@hans](https://github.com/hans)), who wrote [a Python implementation of the model](https://github.com/hans/glove.py/tree/theano) and [a blog post describing that implementation](http://www.foldl.me/2014/glove-python/), which were both very useful references as well.

## References
- Pennington, J., Socher, R., & Manning, C. D. (2014). [Glove: Global vectors for word representation](http://nlp.stanford.edu/pubs/glove.pdf). Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014), 12, 1532-1543.
- [stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe) - Pennington, Socher, and Manning's C implementation of the model
- [hans/glove.py](https://github.com/hans/glove.py) - Jon Gauthier's Python implementation
- [A GloVe implementation in Python](http://www.foldl.me/2014/glove-python/) - Jon's blog post on the topic
