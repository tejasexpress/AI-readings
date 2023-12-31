# Overview
![Pasted image 20230920190956.png](../Images/Pasted%20image%2020230920190956.png)
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
	- <span style="color:#ffc000">Self Attention</span>: Tells which part of input to focus on, for every word in the sentence we can have an attention vector to tell which pairs of words to focus on (how much each word is related to each other word of the same sentence) ![Pasted image 20230920191806.png](../Images/Pasted%20image%2020230920191806.png)
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

>Encode words into numbers
>Encode positions of the words
>Encode relationships among these words

- create three vectors from each of the encoder’s input vectors So for each word, we create a <span style="color:#ffc000">Query vector</span> (what am I looking for) , a <span style="color:#ffc000">Key vector</span> (What can I offer), and a <span style="color:#ffc000">Value vector</span> (What I actually offer). These vectors are created by multiplying the embedding by three matrices that we train during the training process.
- The **second step** in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position. The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2. ![Pasted image 20230921011033.png](../Images/Pasted%20image%2020230921011033.png)
- The **third and fourth steps** are to normalise the scores and pass them through a softmax function so that they turn into a probability distribution. This can be thought of as an attention filter with a value of attention for each pairs of words in an nxn matrix (n = number of words)
- The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example i.e multiplying words with an attention filter telling them which pairs to pay more attention to). 
- The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).![](../Images/Pasted%20image%2020230921160304.png)

> Our goal at the end of the attention layer is to output a set of embeddings 
> for each input word such that the embedding also takes into account the context of the input sentence. i.e it's interaction with the words around it and how much another word in that sentence affects the words
> to achieve this we first calculate the attention filter ( intuition mentioned above) from here we apply the filter to the value vector, what this essentially does is, for each word we want to calculate, it is the <span style="color:#ffc000">attention weighted sum of the value vectors of all the words in the sentence</span> $$ ^{1}V_{new} = \sum_{i=0}^{words}{A_{1i}}^iV_{old}$$
> What this basically means is that it takes the vector of every word, multiplies it by how much it affects the query word ($^1V$) and sums it up. This incorporates the relation between words of a sentence inside the new value vector. $A_{1j}$ represents the first row of the attention filter where the words corresponding to $^1V$ is the query vector and all rest words in the sentence are key vectors. we take the dot products of every key vector with the query vector, put it in a row vector of size $1 *j$ , normalize it and pass it through a soft-max giving us $A_{1j}$
> 
> Note that attention here can mean any semantic relation between the given pair of words that is inferred from the training set. we let backprop decide what type of relations it will amplify.

- The above architecture stacks the self attention blocks parallelly with different initializations for weights which allows the model to capture multiple relationships between the words
- for a masked multi-head attention unit, we mask the upper triangular matrix of the attention filter with -infinity values such that during training, the output words cannot see the words ahead of them but only behind.
- We add and normalize the position encoded input words to the output of the self attention layer to make it easier to train. This happens because the self attention layer can establish relationships among the input words without having to preserve the word embedding information (stronger positional encoding signal). These are called residual connections.
# Decoder

- In the decoder block, we pass in a start token and an end token, and for the output, we shift the words to the left by one i.e given the start token, we want to predict the first word of the output sentence that we passed, with the first word inputted, we want to predict the second word ( of the same sentence)
- we also have two attention block
	- one of this is <span style="color:#ffc000">masked</span> i.e the upper triangular matrix of the attention filter is set to 0, this means the the words can pay 0 attention to the words which come after it. This attention block finds sematic relations between each words and all the words coming before it and outputs words in the <span style="color:#ffc000">second language</span> with numerical embedding + positional encoding (same values as encoder)+ attention context
	- the next attention block takes in the processed words from both the languages and tries to find which words from input languages have sematic relations with words from output languages
		- We create a query vector for the \<SOS> token in the decoder and create keys for each word coming from the encoder and calculate similarities and run it through soft-max to find a distribution of what the decoder thinks will be the next word (after the \<SOS> Token) from this we make a value vector for each word coming from the encoder and multiply it with the corresponding soft-max probability for that word, and then add them to get the encoder-decoder attention values. Stack these parallelly
		- add residual connections to this encoder-decoder attention unit, connect to the output from the decoder self attention unit, we do this so that the network doesn't need to remember then self attention and position encoding values for the decoder
		- This value represents the encoder decoder attention value for the \<SOS> token i.e for that token in decoder, which word from the encoder should he pay the most attention to, to predict the next word coming after the \<SOS> Token (the word from the output language now also has some info from input language encoded within it)
		- We pass this though a fully connected layer which inputs the value for \<SOS> token and outputs = vocabulary in decoder language. we pass it through a soft-max to predict the word most probable after the \<SOS> Token
		- This decoder doesn't stop until it outputs an \<EOS> Token
		- weights for queries, keys and values are different from self attention
