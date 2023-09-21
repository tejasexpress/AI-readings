# Overview
![[Pasted image 20230920190956.png]]
## Why Transformers and not RNNs?

- RNNs are slow to train
- Long sequence lead to vanishing or exploding gradient
## Why Transformers and not LSTMs?

- Better than RNNs
- Slower to train
- Allows memory to retain for longer sequences
## Why Transformers

- Allows parallelization by not requiring to input sequential data one time step at a time, the input data can be passed all at once at the same time. ex- in a translating problem, all words are passed at once 
## Brief Overview of How Transformers Work

- We convert words into <span style="color:#ffc000">input embeddings</span> (vector) such that similar words are physically closer to each other  ( see [[king - man + woman is queen; but why Word Embeddings]] ) 
- <span style="color:#ffc000">Positional Encoder</span>: Vector that gives context based on position of word in the sentence
	- Word embedding + positional encoding = word embedding with context info
	- Positional encoding involves using a sine and cosine function to create a $d$ dimensional vector (same as input embedding) to represent a specific position in a sentence. This vector is <span style="color:#ffc000">not a part of the model itself</span> but is instead added to the input data to provide the model with information about the position of each word in the sentence.
- <span style="color:#ffc000">Encoder:</span>
	- <span style="color:#ffc000">Self Attention</span>: Tells which part of input to focus on, for every word in the sentence we can have an attention vector to tell which pairs of words to focus on (how much each word is related to each other word of the same sentence) ![[Pasted image 20230920191806.png]]
	- <span style="color:#ffc000">Feed Forward</span>: Simple feed forward network applied to every words' attention vector
- <span style="color:#ffc000">Decoder</span>:
	- similar self attention block as encoder by for the output language 
		- Except attention with respect to other words coming before that word is considered such that the network cannot look directly at the next word from that sentence in the training set
	- Next attention block takes input both input language attention vectors and output language attention vector. Here the input lang - output lang word mapping happens
	- gives attention vector for each word with each other word in both the languages
	- Feed forward applied to every attention vector
	- Linear layer is another feed forward neural net with neuron = number of words in output language
	- passed through a softmax function to convert into probability distribution
	- given an input in the decoder block, the highest probability word represents the next predicted word in output language
# Self Attention

- works by seeing how similar each word is to all of the other words (maybe cosine similarity), including itself. i.e if you see a lot of sentences about pizza and the word 'it' was more commonly associated with pizza than oven, then the similarity score for pizza will have a larger impact on how the word it is encoded by the transformer
### Calculating Self Attention

- create three vectors from each of the encoder’s input vectors So for each word, we create a <span style="color:#ffc000">Query vector</span> (what am I looking for) , a <span style="color:#ffc000">Key vector</span> (What can I offer), and a <span style="color:#ffc000">Value vector</span> (What I actually offer). These vectors are created by multiplying the embedding by three matrices that we train during the training process.
- The **second step** in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position. The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2. ![[Pasted image 20230921011033.png]]
- The **third and fourth steps** are to normalise the scores and pass them through a softmax function so that they turn into a probability distribution. This can be thought of as an attention filter with a value of attention for each pairs of words in an nxn matrix (n = number of words)
- The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example). 
- The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).