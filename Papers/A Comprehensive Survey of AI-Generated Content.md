- AIGC is achieved by extracting and understanding intent information from instructions provided by human, and generating the content according to its knowledge and the intent information. In recent years, large-scale models have become increasingly important in AIGC as they provide better intent extraction and thus, improved generation results. With the growth of data and the size of the models, the distribution that the model can learn becomes more comprehensive and closer to reality, leading to more realistic and high-quality content generation![](../Images/Pasted%20image%2020231002160133.png)
- The core advancements in recent AIGC compared to prior works are the result of training more sophisticated generative models on larger datasets, using larger foundation model architectures, and having access to extensive computational resources. For example, the main framework of GPT-3 maintains the same as GPT-2, but the pre-training data size grows from WebText to CommonCrawl, and the foundation model size grows from 1.5B to 175B. Therefore, GPT-3 has better generalization ability than GPT-2 on various tasks, such as human intent extraction.
- In addition to the benefits brought by the increase in data volume and computational power, researchers are also exploring ways to integrate new technologies with GAI algorithms. For example, ChatGPT utilizes reinforcement learning from human feedback (RLHF) to determine the most appropriate response for a given instruction, thus improving model’s reliability and accuracy over time. This approach allows ChatGPT to better understand human preferences in long dialogues.
- Unlike prior methods, generative diffusion models can help generate high-resolution images by controlling the trade-off between exploration and exploitation, resulting in a harmonious combination of diversity in the generated images and similarity to the training data
## History of Generative AI

- In <span style="color:#ffc000">natural language processing</span> (NLP), a traditional method to generate sentences is to learn word distribution using N-gram language modeling and then search for the best sequence. However, this method cannot effectively adapt to long sentences.
- To solve this problem, recurrent neural networks (RNNs) were later introduced for language modeling tasks , allowing for modeling relatively long dependency.
- This was followed by the development of Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), which leveraged gating mechanism to control memory during training. These methods are capable of attending to around 200 tokens in a sample, which marks a significant improvement compared to N-gram language models
- Meanwhile, in <span style="color:#ffc000">computer vision</span> (CV), before the advent of deep learning-based methods, traditional image generation algorithms used techniques such as texture synthesis and texture mapping. These algorithms were based on hand-designed features, and were limited in their ability to generate complex and diverse images.
- In 2014, Generative Adversarial Networks (GANs) was first proposed, which was a significant milestone in this area, due to its impressive results in various applications.
- Variational Autoencoders (VAEs) and other methods like diffusion generative models have also been developed for more fine-grained control over the image generation process and the ability to generate high-quality images
- The advancement of generative models in various domains has followed different paths, but eventually, the intersection emerged: the <span style="color:#ffc000">transformer</span> architecture
- Transformer has later been applied in CV and then become the dominant backbone for many generative models in various domains. In the field of NLP, many prominent large language models, e.g., BERT and GPT, adopt the transformer architecture as their primary building block, offering advantages over previous building blocks, i.e., LSTM and GRU.
- Except for the improvement that transformer brought to individual modalities, this intersection also enabled models from different domains to be fused together for multimodal tasks.
	- One such example of multimodal models is CLIP. CLIP is a joint vision-language model that combines the transformer architecture with visual components, allowing it to be trained on a massive amount of text and image data. Since it combines visual and language knowledge during pre-training, it can also be used as image encoders in multimodal prompting for generation. In all, the emergence of transformer based models revolutionized AI generation and led to the possibility of large-scale training
- In recent years, researchers have also begun to introduce new techniques based on these models. For instance, in NLP, instead of fine-tuning, people sometimes prefer few-shot prompting. which refers to including a few examples selected from the dataset in the prompt, to help the model better understand task requirements![](../Images/Pasted%20image%2020231002161418.png)
## Foundations of AIGC

- <span style="color:#ffc000">Transformer</span> - 
	- Transformer is the backbone architecture for many state-of-the-art models, such as GPT-3, DALL-E-2, Codex, and Gopher It was first proposed to solve the limitations of traditional models such as RNNs in handling variable-length sequences and context awareness.
	- Transformer architecture is mainly based on a self-attention mechanism that allows the model to attend to different parts in a input sequence. Transformer consists of an encoder and a decoder. The encoder takes in the input sequence and generates hidden representations, while the decoder takes in the hidden representation and generates output sequence. Each layer of the encoder and decoder consists of a multi-head attention and a feed-forward neural network.
	- The multi-head attention is the core component of Transformer, which learns to assign different weights to tokens according their relevance. This information routing method allows the model to be better at handling long term dependency, hence, improving the performance in a wide range of NLP tasks.
	- Another advantage of transformer is that its architecture makes it highly parallelizable, and allows data to trump inductive biases. This property makes transformer well-suited for large-scale pre-training, enabling transformer based models to become adaptable to different downstream tasks.
- <span style="color:#ffc000">Pre-trained Language Models</span> - 
	- Since the introduction of the Transformer architecture, it has become the dominant choice in natural language processing due to its parallelism and learning capabilities. Generally, these transformer based pre-trained language models can be commonly classified into two types based on their training tasks: autoregressive language modeling and masked language modeling
	- Given a sentence, which is composed of several tokens, the objective of masked language modeling, e.g., BERT and RoBERTa, refers to predicting the probability of a masked token given context information.![](../Images/Pasted%20image%2020231002164658.png)
	- While autoregressive language modeling, e.g., GPT-3 and OPT, is to model the probability of the next token given previous tokens, hence, left-to-right language modeling. Different from masked language models, autoregressive models are more suitable for generative tasks
- <span style="color:#ffc000">Reinforcement Learning from Human Feedback</span> -
	- Despite being trained on large-scale data, the AIGC may not always produce output that aligns with the user’s intent, which includes considerations of usefulness and truthfulness. In order to better align AIGC output with human preferences, reinforcement learning from human feedback (RLHF) has been applied to fine-tune models in various applications such as Sparrow, InstructGPT, and ChatGPT. Typically, the whole pipeline of RLHF includes the following three steps: pre-training, reward learning, and fine-tuning with reinforcement learning.
	- First, a language model $𝜃_0$ is pre-trained on large-scale datasets as an initial language model.
	- Since the (prompt-answer) pair given by $𝜃_0$ might not align with human purposes, in the second step we train a reward model to encode the diversified and complex human preference. Specifically, given the same prompt 𝑥, different generated answers $\{𝑦_1, 𝑦_2, · · · , 𝑦_3\}$ are evaluated by humans in a pairwise manner. The pairwise comparison relationships are later transferred to pointwise reward scalars, $\{𝑟_1, 𝑟_2, · · · , 𝑟_3\}$, using an algorithm such as ELO
	- In the final step, the language model $𝜃$ is fine-tuned to maximize the learned reward function using reinforcement learning. To stabilize the RL training, Proximal Policy Optimization (PPO) is often used as the RL algorithm.
	- Although RLHF has shown promising results by incorporating fluency, progress in this field is impeded by a lack of publicly available benchmarks and implementation resources, leading to a perception that RL is a challenging approach for NLP. To address this issue, an open-source library named RL4LMs has recently been introduced, consisting of building blocks for fine-tuning and evaluating RL algorithms on LM-based generation.
- <span style="color:#ffc000">Generative Language Models</span> - 
	- Recently, The use of pre-trained language models has emerged as the prevailing technique in the domain of NLP. Generally, current state-of-the-art pre-trained language models could be categorized as masked language models (encoders), autoregressive language models (decoders) and encoder-decoder language models
	- Decoder models are widely used for text generation, while encoder models are mainly applied to classification tasks. By combining the strengths of both structures, encoder-decoder models can leverage both context information and autoregressive properties to improve performance across a variety of tasks. The primary focus of this survey is on generative models.
	- <span style="color:#ffc000">Decoder Models</span> - 
		- One of the most prominent examples of autoregressive decoder-based language models is GPT, which is a transformer-based model that utilizes self-attention mechanisms to process all words in a sequence simultaneously. GPT is trained on next word prediction task based on previous words, allowing it to generate coherent text. 
		- Subsequently, GPT-2 and GPT-3 maintains the autoregressive left-to-right training method, while scaling up model parameters and leveraging diverse datasets beyond basic web text, achieving state-of-the-art results on numerous datasets.![](../Images/Pasted%20image%2020231002201600.png)
		- Except for the advancements in model architecture and pre-training tasks, there has also been significant efforts put into improving the fine-tuning process for language models.
	- <span style="color:#ffc000">Encoder-Decoder Models</span> - 
		- One of the main encoder-decoder methods is Text-to-Text Transfer Transformer which combines transformer-based encoders and decoders together for pre-training. T5 employs a "text-to-text" approach, which means that it transforms both the input and output data into a standardized text format. This allows T5 to be trained on a wide range of NLP tasks, such as machine translation, question-answering, summarization, and more, using the same model architecture.
- <span style="color:#ffc000">Vision Generative Models</span> - 
	- <span style="color:#ffc000">Generative Adversarial Networks (GANs)</span> - 