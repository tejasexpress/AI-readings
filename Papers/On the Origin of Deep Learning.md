Author: Haohan Wang, Bhiksha Raj
Link to paper: https://arxiv.org/pdf/1702.07800.pdf

One remarkable property of neural networks, widely known as universal approximation property, roughly describes that an MLP can represent any functions. Here we discussed this property in three different aspects: 

• Boolean Approximation: an MLP of one hidden layer1 can represent any boolean function exactly. 

• Continuous Approximation: an MLP of one hidden layer can approximate any bounded continuous function with arbitrary accuracy. 

• Arbitrary Approximation: an MLP of two hidden layers can approximate any function with arbitrary accuracy.

---

here are multiple reasons why DNNs sparked when they did (stars had to align, like all things similar, it's just the matter of right place, right time etc).

One reason is the availability of data, lots of data (labeled data). If you want to be able to generalize and learn something like 'generic priors' or 'universal priors' (aka the basic building blocks that can be re-used between tasks / applications) then you need lots of data. And wild data, might I add, not sterile data-sets carefully recorded in the lab with controlled lighting and all. Mechanical Turk made that (labeling) possible.

Second, the possibility to train larger networks faster using GPUs made experimentation faster. ReLU units made things computationally faster as well and provided their regularization since you needed to use more units in one layer to be able to compress the same information since layers now were more sparse, so it also went nice with dropout. Also, they helped with an important problem that happens when you stack multiple layers. More about that later. Various multiple tricks that improved performance. Like using mini-batches (which is in fact detrimental for final error) or convolutions (which actually don't capture as much variance as local receptive fields) but are computationally faster.

In the meantime people were debating if they liked em more skinny or more chubby, smaller or taller, with or without freckles, etc. Optimization was like does it fizz or does it bang so research was moving towards more complex methods of training like conjugate gradient and newtons method, finally they all realized there's no free lunch. Networks were burping.

What slowed things down was the *vanishing gradient* problem. People went like: whoa, that's far out, man! In a nutshell it means that it was hard to adjust the error on layers closer to the inputs. As you add more layers on the cake, gets too wobbly. You couldn't back-propagate meaningful error back to the first layers. The more layers, the worse it got. Bummer.

Some people figured out that using the cross-entropy as a loss function (well, again, classification and image recognition) provides some sort of regularization and helps against the network getting saturated and in turn the gradient wasn't able to hide that well.

---

**The Necessity of Depth**

The universal approximation properties of shallow neural networks come at a price of exponentially many neurons and therefore are not realistic. The question about how to maintain this expressive power of the network while reducing the number of computation units has been asked for years. Intuitively, Bengio and Delalleau (2011) suggested that it is nature to pursue deeper networks because 
1) human neural system is a deep architecture (as we will see examples in Section 5 about human visual cortex.) and 
2) humans tend to represent concepts at one level of abstraction as the composition of concepts at lower levels

This conclusion could trace back to three decades ago when Yao (1985) showed the limitations of shallow circuits functions. Hastad (1986) later showed this property with parity circuits: “there are functions computable in polynomial size and depth k but requires exponential size when depth is restricted to k − 1”. He showed this property mainly by the application of DeMorgan’s law, which states that any AND or ORs can be rewritten as OR of ANDs and vice versa. Therefore, he simplified a circuit where ANDs and ORs appear one after the other by rewriting one layer of ANDs into ORs and therefore merge this operation to its neighboring layer of ORs. By repeating this procedure, he was able to represent the same function with fewer layers, but more computations.

Moving from circuits to neural networks, Delalleau and Bengio (2011) compared deep and shallow sum-product neural networks. They showed that a function that could be expressed with O(n) neurons on a network of depth k required at least O(2 √ n ) and O((n − 1)k ) neurons on a two-layer neural network.

They showed that for a shallow network, the representation power can only grow polynomially with respect to the number of neurons, but for deep architecture, the representation can grow exponentially with respect to the number of neurons.

However, in reality, many problems will arise if we keep increasing the layers. Among them, the increased difficulty of learning proper parameters is probably the most prominent one.

---

The breakthrough ResNet introduces, which allows ResNet to be substantially deeper than previous networks, is called Residual Block. The idea behind a Residual Block is that some input of a certain layer (denoted as x) can be passed to the component two layers later either following the traditional path which involves convolutional layers and ReLU transform succession (we denote the result as f(x)), or going through an express way that directly passes x there. As a result, the input to the component two layers later is f(x) + x instead of what is typically seen as f(x).![Pasted image 20230920184810.png](Pasted%20image%2020230920184810.png)
Another interesting perspective of ResNet is provided by (Veit et al., 2016). They showed that ResNet behaves like ensemble of shallow networks: the express way introduced allows ResNet to perform as a collection of independent networks, each network is significantly shallower than the integrated ResNet itself. This also explains why gradient can be passed through the ultra-deep architecture without being vanished

In addition, the idea of Residual Block has been found in the actual visual cortex (In the ventral stream of the visual cortex, V4 can directly accept signals from primary visual cortex), although ResNet is not designed according to this in the first place.

---

**Network Property and Vision Blindness Spot**

- **Adversarial examples**
    - Szegedy et al. (2013) showed that they could force a deep learning model to misclassify an image simply by adding perturbations to that image. More importantly, these perturbations may not even be observed by naked human eyes. In other words, two objects that look almost the same to human, may be recognized as different objects by a well-trained neural network (for example, AlexNet). They have also shown that this property is more likely to be a modeling problem, in contrast to problems raised by insufficient training. On the other hand, Nguyen et al. (2015) showed that they could generate patterns that convey almost no information to human, being recognized as some objects by neural networks with high confidence (sometimes more than 99%). Since neural networks are typically forced to make a prediction, it is not surprising to see a network classify a meaningless patter into something, however, this high confidence may indicate that the fundamental differences between how neural networks and human learn to know this world.
    - With construction, we can show that the neural networks may misclassify an object, which should be easily recognized by the human, to something unusual. On the other hand, a neural network may also classify some weird patterns, which are not believed to be objects by the human, to something we are familiar with. Both of these properties may restrict the usage of deep learning to real world applications when a reliable prediction is necessary
    - Even without these examples, one may also realize that the reliable prediction of neural networks could be an issue due to the fundamental property of a matrix: the existence of null space. As long as the perturbation happens within the null space of a matrix, one may be able to alter an image dramatically while the neural network still makes the misclassification with high confidence. Null space works like a blind spot to a matrix and changes within null space are never sensible to the corresponding matrix
- **Human labelling**
    - To further improve the performance ResNet reached, one direction might be to modeling the annotators’ labeling preference. One assumption could be that annotators prefer to label an image to make it distinguishable. Some established work to modeling human factors (Wilson et al., 2015) could be helpful. However, the more important question is that whether it is worth optimizing the model to increase the testing results on ImageNet dataset, since remaining misclassifications may not be a result of the incompetency of the model, but problems of annotations

---

**Rprop**

RpropRprop was introduced by Riedmiller and Braun (1993). It is a unique method even studied
back today as it does not fully utilize the information of gradient, but only considers the
sign of it.![Pasted image 20230920184741.png](Pasted%20image%2020230920184741.png)
**Dropout**

Dropout was introduced in (Hinton et al., 2012; Srivastava et al., 2014). The technique soon got influential, not only because of its good performance but also because of its simplicity of implementation. The idea is very simple: randomly dropping out some of the units while training. More formally: on each training case, each hidden unit is randomly omitted from the network with a probability of p. As suggested by Hinton et al. (2012), Dropout can be seen as an efficient way to perform model averaging across a large number of different neural networks, where overfitting can be avoided with much less cost of computation.

The main motivation behind the algorithm is to prevent the co-adaptation of feature detectors, or
overfitting, by forcing neurons to be robust and rely on population behavior, rather than on the
activity of other specific units. In spite of the impressive results that have been reported, little is known about dropout from a theoretical standpoint, in particular about its averaging, regularization, and convergence properties.