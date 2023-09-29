Deep-Learning-based Software Engineering
Multi-Agent Collaboration
*** 
![](../Images/Pasted%20image%2020230929144011.png)

- Software engineering is a domain characterized by intricate decision-making processes, often relying on nuanced intuition and consultation. At the core of this paradigm lies CHATDEV, a virtual chat-powered software development company that mirrors the established waterfall model, meticulously dividing the development process into four distinct chronological stages: designing, coding, testing, and documenting. Each stage engages a team of agents, such as programmers, code reviewers, and test engineers, fostering collaborative dialogue and facilitating a seamless workflow. The instrumental analysis of CHATDEV highlights its remarkable efficacy in software generation, enabling the completion of the entire software development process in under seven minutes at a cost of less than one dollar. It not only identifies and alleviates potential vulnerabilities but also rectifies potential hallucinations while maintaining commendable efficiency and cost-effectiveness.
- (see [The Rise and Potential of Large Language Model Based Agents](The%20Rise%20and%20Potential%20of%20Large%20Language%20Model%20Based%20Agents.md))
## Introduction

- The software development process involves multiple roles, including organizational coordination, task allocation, code writing, system testing, and documentation preparation. It is a highly complex and intricate activity that demands meticulous attention to detail due to its long development cycles
- After training on massive corpora using the “next word prediction” paradigm, LLMs have demonstrated impressive performance on a wide range of downstream tasks, such as context-aware question answering, machine translation, and code generation. In fact, the core elements involved in software development, namely codes and documents, can both be regarded as “language” (i.e., sequences of characters). From this perspective, this paper explores an end-to-end software development framework driven by LLMs, encompassing requirements analysis, code development, system testing, and document generation, aiming to provide a unified, efficient, and cost-effective paradigm for software development.
- Directly generating an entire software system using LLMs can result in code hallucinations to a certain extent, similar to the phenomenon of hallucination in natural language knowledge querying. These hallucinations include incomplete implementation of functions, missing dependencies, and potential undiscovered bugs. Code hallucinations arise primarily due to two reasons.
	- the lack of task specificity confuses LLMs when generating a software system in one step. Granular tasks in software development, such as analyzing user/client requirements and selecting programming languages, provide guided thinking that is absent in the high-level nature of the task handled by LLMs.
	- the absence of cross-examination in decision-making poses significant risks
- It follows the classic waterfall model and divides the process into four phases: designing, coding, testing, and documenting. At each phase, CHATDEV recruits multiple agents with different roles, such as programmers, reviewers, and testers. To facilitate effective communication and collaboration, CHATDEV utilizes a proposed chat chain that divides each phase into atomic subtasks. 
	- Within the chat chain, each node represents a specific subtask, and two roles engage in context-aware, multi-turn discussions to propose and validate solutions.
	- This approach ensures that client requirements are analyzed, creative ideas are generated, prototype systems are designed and implemented, potential issues are identified and addressed, debug information is explained, appealing graphics are created, and user manuals are generated. By guiding the software development process along the chat chain, CHATDEV delivers the final software to the user, including source code, dependency environment specifications, and user manuals.
- Discussions between a reviewer and a programmer led to the identification and modification of nearly twenty types of code vulnerabilities, while discussions between a tester and a programmer resulted in the identification and resolution of more than ten types of potential bugs.
- <span style="color:#ffc000">Main Contributions</span> - 
	- We propose CHATDEV, a chat-based software development framework. By merely specifying a task, CHATDEV sequentially handles designing, coding, testing, and documenting. This new paradigm simplifies software development by unifying main processes through language communication, eliminating the need for specialized models at each phase.
	- We propose the chat chain to decompose the development process into sequential atomic subtasks. Each subtask requires collaborative interaction and cross-examination between two roles. This framework enables multi-agent collaboration, user inspection of intermediate outputs, error diagnoses, and reasoning intervention. It ensures a granular focus on specific subtasks within each chat, facilitating effective collaboration and promoting the achievement of desired outputs.
	- To further alleviate potential challenges related to code hallucinations, we introduce the thought instruction mechanism in each independent chat process during code completion, reviewing, and testing. By performing a “role flip”, an instructor explicitly injects specific thoughts for code modifications into instructions, thereby guiding the assistant programmer more precisely.
	- The experiments demonstrate the efficiency and cost-effectiveness of CHATDEV’s automated software development process. Through effective communication, proposal, and mutual examination between roles in each chat, the framework enables effective decision-making
## ChatDev

- Similar to hallucinations encountered when using LLMs for natural language querying, directly generating entire software systems using LLMs can result in severe code hallucinations, such as incomplete implementation, missing dependencies, and undiscovered bugs. These hallucinations may stem from the lack of specificity in the task and the absence of cross-examination in decision making.
- When presented with a task, the diverse agents at CHATDEV collaborate to develop a required software, including an executable system, environmental guidelines, and user manuals. This paradigm revolves around leveraging large language models as the core thinking component, enabling the agents to simulate the entire software development process, circumventing the need for additional model training and mitigating undesirable code hallucinations to some extent.
### ChatChain

- CHATDEV employs the widely adopted waterfall model, a prominent software development life cycle model, to divide the software development process into four distinct phases:
	- <span style="color:#ffc000">designing</span> - innovative ideas are generated through collaborative brainstorming, and technical design requirements are defined
	- <span style="color:#ffc000">coding</span> - Involves the development and review of source code
	- <span style="color:#ffc000">testing</span> - integrates all components into a system and utilizes feedback messages from interpreter for debugging
	- <span style="color:#ffc000">documenting</span> - encompasses the generation of environment specifications and user manuals
- Each of these phases necessitates effective communication among multiple roles, posing challenges in determining the sequence of interactions and identifying the relevant individuals to engage with.
- To address this, we propose a generalized architecture by breaking down each phase into multiple atomic chats, each with a specific focus on task-oriented role-playing involving two distinct roles.![](../Images/Pasted%20image%2020230929151627.png)
- In each chat, an instructor initiates instructions, guiding the dialogue towards task completion, while the assistant follows the instructions, provides suitable solutions, and engages in discussions regarding feasibility. The instructor and assistant cooperate through multi-turn dialogues until they reach a consensus and determine that the task has been successfully accomplished.
- The chat chain provides a transparent view of the software development process, shedding light on the decision-making path and offering opportunities for debugging when errors arise, which enables users to inspect intermediate outputs, diagnose errors, and intervene in the reasoning process if necessary.
- Besides, chat chain ensures a granular focus on specific subtasks within each phase, facilitating effective collaboration and promoting the attainment of desired outputs.
### Designing
- In the designing phase, CHATDEV receives an initial idea from a human client. This phase involves three predefined roles: CEO (chief executive officer), CPO (chief product officer), and CTO (chief technology officer). The chat chain then breaks down the designing phase into sequential atomic chatting tasks, including decisions regarding the target software’s modality (CEO and CPO) and the programming language (CEO and CTO)![](../Images/Pasted%20image%2020230929155446.png)
- <span style="color:#ffc000">Role Assignment</span> - System prompts/messages are used to assign roles to each agent during the role-playing process. In contrast to other conversational language models, our approach to prompt engineering is restricted solely to the initiation of role-playing scenarios. The instructor is denoted as $P_I$, while the assistant’s system prompt/message is denoted as $P_A$. These prompts assign roles to the agents before the dialogues begin. Let LI and LA represent two large language models. Using the system message, we have $I ← L^{P_I}_I$ and $A ← L^{P_A}_A$ , which serve as the instructor and assistant agents
	- In our framework, the instructor initially acts as a CEO, engaging in interactive planning, while the assistant assumes the role of CPO, executing tasks and providing responses. To achieve role specialization, we employ <span style="color:#ffc000">inception prompting</span>, which has proven effective in enabling agents to fulfill their roles. The instructor and assistant prompts encompass vital details concerning the designated task and roles, communication protocols, termination criteria, and constraints aimed at preventing undesirable behaviors (e.g., instruction redundancy, uninformative responses, infinite loops, etc.).
- <span style="color:#ffc000">Memory Stream</span> - The memory stream is a mechanism that maintains a comprehensive record of an agent’s previous dialogues, assisting in subsequent decision-making in an utterance-aware manner. We establish communication protocols through prompts. For example, an ending message satisfying specific formatting requirements (e.g., “MODALITY: Desktop Application”) is generated when both parties reach a consensus. The system monitors communication to ensure compliance with the designated format, allowing for the conclusion of the current dialogue.
- <span style="color:#ffc000">Self-Reflection</span> - Occasionally, we have observed dialogues where both parties reach a consensus but do not trigger the predefined communication protocols as termination conditions. we introduce a self-reflection mechanism, which involves extracting and retrieving memories. To implement this mechanism, we enlist a “pseudo self” as a new questioner and initiate a fresh chat. The pseudo questioner informs the current assistant of all the historical records from previous dialogues and requests a summary of the conclusive information from the dialogue![](../Images/Pasted%20image%2020230929162004.png)
### Coding
- The coding phase involves three predefined roles: CTO, programmer, and art designer. The chat chain decomposes the coding phase into sequential atomic chatting tasks, such as generating complete codes (CTO and programmer) and devising a graphical user interface (designer and programmer). Based on the main designs discussed in the previous phase, the CTO instructs the programmer to implement a software system using markdown format.
- The programmer generates codes in response and extracts the corresponding codes based on markdown format. The designer proposes a user-friendly graphical user interface (GUI) that uses graphical icons for user interaction instead of text-based commands. Then, the designer creates visually appealing graphics using external text-to-image tools, which the programmer incorporates into the GUI design using standard toolkits.
- <span style="color:#ffc000">Code Management</span> - To handle complex software systems, CHATDEV utilizes object-oriented programming languages like Python. The modularity of object-oriented programming allows for selfcontained objects, aiding troubleshooting and collaborative development. Reusability enables code reuse through inheritance, reducing redundancy. We introduce the “version evolution” mechanism to restrict visibility to the latest code version between roles, discarding earlier code versions from the memory stream. The programmer manages the project using Git-related commands. Proposed code modifications and changes increment the software version by 1.0. Version evolution gradually eliminates code hallucinations. The combination of object-oriented programming and version evolution is suitable for dialogues involving long code segments.
- <span style="color:#ffc000">Thought Instruction</span> - Traditional question answering can lead to inaccuracies or irrelevant information, especially in code generation, where naive instructions may result in unexpected hallucinations. This issue becomes particularly problematic when generating code. For instance, when instructing the programmer to implement all unimplemented methods, a naive instruction may result in hallucinations, such as including methods that are reserved as unimplemented interfaces. To address this, we propose the “thought instruction” mechanism, inspired by chain-of-thought prompting.
	- It involves explicitly addressing specific problem-solving thoughts in instructions, akin to solving subtasks in a sequential manner. thought instruction includes swapping roles to inquire about which methods are not yet implemented and then switching back to provide the programmer with more precise instructions to follow. By incorporating thought instruction, the coding process becomes more focused and targeted. The explicit expression of specific thoughts in the instructions helps to reduce ambiguity and ensures that the generated code aligns with the intended objectives. This mechanism enables a more accurate and context-aware approach to code completion, minimizing the occurrence of hallucination and resulting in more reliable and comprehensive code outputs.
### Testing
- Even for human programmers, there is no guarantee that the code they write on the first attempt is always error-free. Rather than discarding incorrect code outright, humans typically analyze and investigate code execution results to identify and rectify implementation errors
- In CHATDEV, the testing phase involves three roles: programmer, reviewer, and tester. The process consists of sequential atomic chatting tasks, including peer review (programmer and reviewer) and system testing (programmer and tester).
	- Peer review, or static debugging, examines source code to identify potential issues.
	- System testing, a form of dynamic debugging, verifies software execution through tests conducted by the programmer using an interpreter.
	- This testing focuses on evaluating application performance through black-box testing.
- In our practice, we observed that allowing two agents to communicate solely based on feedback messages from an interpreter does not result in a bug-free system. The programmer’s modifications may not strictly follow the feedback, leading to hallucinations. To address this, we further employ the thought instruction mechanism to explicitly express debugging thoughts in the instructions
- The tester executes the software, analyzes bugs, proposes modifications, and instructs the programmer accordingly. This iterative process continues until potential bugs are eliminated and the system runs successfully.
- In cases where an interpreter struggles with identifying fine-grained logical issues, the involvement of a human client in software testing becomes optional.
### Documenting
- After the designing, coding, and testing phases, CHATDEV employs four agents (CEO, CPO, CTO, and programmer) to generate software project documentation. Using large language models, we leverage few-shot prompting with in-context examples for document generation. The CTO instructs the programmer to provide configuration instructions for environmental dependencies, resulting in a document like requirements.txt. This document allows users to configure the environment independently. Simultaneously, the CEO communicates requirements and system design to the CPO, who generates a user manual. ![](../Images/Pasted%20image%2020230929163631.png)
## Experimentation and Results

- a small fraction, 13.33% of the software, encountered execution failures. Upon analyzing the failed software creations, we identified two primary contributing factors. 
	- Firstly, in 50% of the cases, the failure was attributed to the token length limit of the API. This limitation prevented obtaining the complete source code within the specified length constraint for code generation. Such challenges are particularly evident when dealing with complex software systems or scenarios requiring extensive code generation.
	- The remaining 50% of the failed software creations were primarily affected by external dependency issues. These challenges emerged when certain dependencies were either unavailable in the cloud or incorrectly versioned, resulting in conflicts and unavailability of specific application programming interfaces (APIs) in the current version
- The average total cost in software production is approximately $0.15693 . To determine the overall cost of software development with CHATDEV, we also consider the cost of designer-produced images. The average designer cost is $0.1398 per software for each software production involving 8.74 graphics creations on average. Thus, the average software development cost at CHATDEV is $0.2967, significantly lower than traditional custom software development companies’ expenses
- The most frequently discussed issue in the reviewer-programmer communication during code review is “methods not implemented” (34.85%). This challenge commonly arises in code generation for complex models, where core functionalities often receive placeholder labels (such as “pass” in Python) to be further completed. 
- Additionally, the dialogue frequently addresses the topic of “modules not imported” (19.70%). This issue emerges from the nature of code generation, where the generated code tends to overlook minor details.
- the thought instruction mechanism proposed in this paper effectively tackles these issues by compelling the reviewer to identify incomplete methods and requiring the programmer to fill them. <span style="color:#ffc000">This mechanism can be applied to other scenarios where tasks are completed based on large models but with certain parts missing.</span>
- Interestingly, the reviewer also emphasizes the importance of code robustness. They underscore considerations for handling potential exceptions in the future and offer hints on avoiding duplicate categories (3.03%). Additionally, the reviewer provides suggestions regarding unused classes in the code (1.52%), identifies infinite loops (1.52%), and emphasizes the necessity of proper environment initialization (1.52%).![](../Images/Pasted%20image%2020230929164459.png)![](../Images/Pasted%20image%2020230929164545.png)
- As observed in the figure, the most frequent debug issue between the tester and the programmer is “module not found” (45.76%), accounting for nearly half of the cases. This reflects the model’s tendency to overlook very fine details, despite their simplicity. Fortunately, with the thought instruction mechanism proposed in this paper, such bugs can often be easily resolved by importing the required class or method. The second most common types of errors are “attribute error” and “unknown option”, each accounting for 15.25% of the cases. “attribute error” refers to errors in the usage of class attributes, while “unknown option” indicates errors in the parameters of method calls. Another common type of error is “import error” which is similar to “module not found” and is primarily caused by mistakes in the import statements, such as importing the wrong class or using an incorrect import path. In addition to these common error types, CHATDEV has the capability to detect relatively rare error types such as improperly initialized GUI (5.08%), incorrect method calling (1.69%), missing file dependencies (1.69%), unused modules (1.69%), decorator syntax errors (1.69%), and more![](../Images/Pasted%20image%2020230929164855.png)
## Discussion

- Even when we set the temperature parameter of the large language model to a very low value, we observe inherent randomness in the generated output. Consequently, each software produced may vary between different runs. As a result, this technology is best suited for open and creative software production scenarios where variations are acceptable. Moreover, there are instances where the software fails to meet the users’ needs. This can be attributed to unclear user requirements and the inherent randomness in text or code generation.
- While the designer agent is capable of creating images, it is important to acknowledge that the directly generated images may not always enhance the GUI’s aesthetics. At times, they may introduce excessive complexity, which can hinder user experience. This is primarily because each image is generated independently, lacking direct visual correlation. To address this, we have provided the option for users to customize the GUI as a system hyperparameter, allowing them to decide whether to enable this feature or not.
- the generated software currently lacks malicious intent identification for sensitive file operations. Therefore, users are advised to conduct their own code review before running the software to prevent any unnecessary data loss.
- Although the study may potentially help junior programmers or engineers in real world, it is challenging for the system to generate perfect source code for high-level or large-scale software requirements. This difficulty arises from the agents’ limited ability to autonomously determine specific implementation details, often resulting in multiple rounds of lengthy discussions. Additionally, large-scale software development proves challenging for both reviewers and testers, as it becomes difficult to identify defects or vulnerabilities within the given time constraints.
- Moving forward, further research can focus on refining the communication protocols and optimizing the interaction dynamics within each chat to enhance the performance and effectiveness of CHATDEV. Additionally, exploring the integration of other emerging technologies, such as reinforcement learning and explainable AI, could provide valuable insights into addressing challenges and improving the overall software development process.
- The overarching objective is to achieve even greater efficiency in software production by improving various characteristics, such as reducing the length of chat chains or optimizing subtask solving logic and strategies, ultimately leading to more streamlined and effective software production processes.