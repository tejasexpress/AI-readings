Author: Piotr Migdał, Kai Chen, Greg Corrado, Jeffrey Dean,
Tomas Mikolov, CHRIS MOODY, Chris McCormick, Jay Alammar
Link to paper: https://p.migdal.pl/blog/2017/01/king-man-woman-queen-why/
# **What Are Word Embeddings?**

A word embedding is a learned representation for text where words that have the same meaning have a similar representation.

**how do you represent words as vectors such that two similar vectors are two similar words and their direction might also have some meaning**

> *One of the benefits of using dense and low-dimensional vectors is computational: the majority of neural network toolkits do not play well with very high-dimensional, sparse vectors. … The main benefit of the dense representations is generalization power: if we believe some features may provide similar clues, it is worthwhile to provide a representation that is able to capture these similarities.*
> 
- Key to the approach is the idea of using a dense distributed representation for each word.
- Each word is represented by a real-valued vector, often tens or hundreds of dimensions. This is contrasted to the thousands or millions of dimensions required for sparse word representations, such as a one-hot encoding.
- The distributed representation is learned based on the usage of words. This allows words that are used in similar ways to result in having similar representations, naturally capturing their meaning. This can be contrasted with the crisp but fragile representation in a bag of words model where, unless explicitly managed, different words have different representations, regardless of how they are used.
- Instead of choosing these numbers to represent the words manually, we can use a neural network to do it for us.
- The counterpart is that often antonyms are also very close in that same space. That’s how Word2vec works. Words that appear in the same context – and antonyms usually do – are mapped in the same area of space.

# Word2Vec

- word2vec is an algorithm that transforms words into vectors, so that words with similar meaning end up laying close to each other. Moreover, it allows us to use vector arithmetics to work with analogies, for example the famous .`king - man + woman = queen`
- **assuming distributional hypothesis**
    - methods for quantifying and categorizing semantic similarities between linguistic items based on their distributional properties in large samples of language data. The basic idea of distributional semantics can be summed up in the so-called distributional hypothesis: linguistic items with similar distributions have similar meanings.
    - *a word is characterized by the company it keeps*
    - **words that appear in similar contexts are similar words**

# Efficient Estimation of Word Representations in
Vector Space

### Intuition behind why?

- If you have a language model that is able to get good performance of predicting the next word in a sentence and the architecture of that model is such that it doesn’t have that many neurons in it’s hidden layers. It has to be compressing that info down efficiently
- The essence of word2vec & similar embedding models may be **compression**: the model is forced to predict neighbors using **far less internal state** than would be required to remember the entire training set. So it has to force similar words together, in similar areas of the parameter space, and force groups of words into various useful relative-relationships.

### Continuous Bag-of-Words Model

- The first proposed architecture is similar to the feedforward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words
- all words get projected into the same position (their vectors are averaged). the order of words in the history does not influence the projection. Furthermore, we also use words from the future
- All those random/non-predictive instances tend to cancel-out as noise; the relationships that have **some** ability to predict nearby words, even slightly, eventually find **some** relative/nearby arrangement in the high-dimensional space, so as to help the model for some training examples.
- Note that a word2vec model isn't necessarily an **effective** way to predict nearby words. It might never be good at that task. But the **attempt** to become good at neighboring-word prediction, with fewer free parameters than would allow a perfect-lookup against training data, forces the model to **LEARN** underlying semantic or syntactic patterns in the data.

### Continuous Skip-gram Model

- The second architecture is similar to CBOW, but instead of predicting the current word based on the context, it tries to maximize classification of a word based on another word in the same sentence.
- More precisely, we use each current word as an input to a log-linear classifier with continuous
projection layer, and predict words within a certain range before and after the current word. We
found that increasing the range improves quality of the resulting word vectors, but it also increases the computational complexity.
- Since the more distant words are usually less related to the current
word than those close to it, we give less weight to the distant words by sampling less from those words in our training examples.
![Untitled.png](Untitled.png)

- To find a word that is similar to small in the same sense as
biggest is similar to big, we can simply compute vector X = vector(”biggest”)−vector(”big”) +
vector(”small”). Then, we search in the vector space for the word closest to X measured by cosine distance, and use it as the answer to the question (we discard the input question words during this search). When the word vectors are well trained, it is possible to find the correct answer (word smallest) using this method.
- Finally, we found that when we train high dimensional word vectors on a large amount of data, the resulting vectors can be used to answer very subtle semantic relationships between words, such as a city and the country it belongs to, e.g. France is to Paris as Germany is to Berlin.

# A Word is Worth a Thousand Vectors

- We’ve simply fed a mountain of text into an algorithm called word2vec and expected it to learn from context. Word by word, it tries to predict the other surrounding words in a sentence. Or rather, it internally represents words as vectors, and given a word vector, it tries to predict the other word vectors in the nearby text
- The algorithm eventually sees so many examples that it can infer the gender of a single word, that both the The Times and The Sun are newspapers, that The Matrix is a sci-fi movie, and that the style of an article of clothing might be boho or edgy. That word vectors represent much of the information available in a dictionary definition is a convenient and almost miraculous side effect of trying to predict the context of a word.
- Internally high dimensional vectors stand in for the words, and some of those dimensions are encoding gender properties. Each axis of a vector encodes a property, and the magnitude along that axis represents the relevance of that property to the word
- We have the ability to search semantically by adding and subtracting word vectors. This empowers us to creatively add and subtract concepts and ideas.

# ****Word2Vec Tutorial - The Skip-Gram Model****

- Word2Vec uses a trick you may have seen elsewhere in machine learning. We’re going to train a simple neural network with a single hidden layer to perform a certain task, but then we’re not actually going to use that neural network for the task we trained it on! Instead, the goal is actually just to learn the weights of the hidden layer–we’ll see that these weights are actually the “word vectors” that we’re trying to learn.

<aside>
💡 Another place you may have seen this trick is in unsupervised feature learning, where you train an auto-encoder to compress an input vector in the hidden layer, and decompress it back to the original in the output layer. After training it, you strip off the output layer (the decompression step) and just use the hidden layer--it's a trick for learning good image features without having labeled training data.

</aside>

- We’re going to train the neural network to do the following. Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. When I say "nearby", there is actually a "window size" parameter to the algorithm. A typical window size might be 5, meaning 5 words behind and 5 words ahead (10 in total).
- For example, if you gave the trained network the input word “Soviet”, the output probabilities are going to be much higher for words like “Union” and “Russia” than for unrelated words like “watermelon” and “kangaroo”.
- The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.
![untitled 1.png](untitled%201.png)

![untitled 2.png](untitled%202.png)

A few things to point out:

1. There’s a straight red column through all of these different words. They’re similar along that dimension (and we don’t know what each dimensions codes for)
2. You can see how “woman” and “girl” are similar to each other in a lot of places. The same with “man” and “boy”
3. “boy” and “girl” also have places where they are similar to each other, but different from “woman” or “man”. Could these be coding for a vague conception of youth? possible.
4. All but the last word are words representing people. I added an object (water) to show the differences between categories. You can, for example, see that blue column going all the way down and stopping before the embedding for “water”.
5. There are clear places where “king” and “queen” are similar to each other and distinct from all the others. Could these be coding for a vague concept of royalty?

The famous examples that show an incredible property of embeddings is the concept of analogies. We can add and subtract word embeddings and arrive at interesting results.

- It’s important to appreciate that all of these properties of W are *side effects* . We didn’t try to have similar words be close together. We didn’t try to have analogies encoded with difference vectors. All we tried to do was perform a simple task, like predicting whether a sentence was valid. These properties more or less popped out of the optimization process.