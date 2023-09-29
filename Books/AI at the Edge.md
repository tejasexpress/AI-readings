Typically, IoT devices have been viewed as simple nodes that collect data from sensors and then transmit it to a central location for processing. The problem with this approach is that sending such large volumes of low-value information is extraordinarily costly. Not only is connectivity expensive, but transmitting data uses a ton of energy—which is a big problem for battery powered IoT devices. Because of this problem, the vast majority of data collected by IoT sensors has usually been discarded. We’re collecting a ton of sensor data, but we’re unable to do anything with it![](../Images/Pasted%20image%2020230929025608.png)
- When we talk about embedded ML, we’re usually referring to machine learning inference—the process of taking an input and coming up with a prediction (like guessing a physical activity based on accelerometer data). The training part usually still takes place on a conventional computer
- Digital signal processing (DSP) is the practice of using algorithms to manipulate these streams of data. When paired with embedded machine learning, we often use DSP to modify signals before feeding them into machine learning models. DSP is so common for embedded systems that often embedded chips have super fast hardware implementations of common DSP algorithms, just in case you need them.
- TinyML4D
- A lot of AI applications are powered by machine learning. Most of the time, machine learning involves training a model to make predictions based on a set of labeled data. Once the model has been trained, it can be used for inference: making new predictions on data it has not seen before.
	- There are two subtypes of on-device training that are more widespread. One of these is used commonly in tasks such as facial or fingerprint verification on mobile phones, to map a set of biometrics to a particular user. The second is used in predictive maintenance, where an on-device algorithm learns a machine’s “normal” state so that it can act if the state becomes abnormal.
- The goal of an edge AI deployment is to make sense of this data, identifying patterns and using them to make decisions. This is a major challenge, especially given that most embedded devices are resource-constrained and don’t have the RAM to store large amounts of data.
- The goal is to build applications that make the most of this “good enough” performance
- imagine a basic fitness wearable that uses an accelerometer to count steps. Even this simple device might be equipped with a sensitive multiaxis accelerometer that has a very high sample rate, capable of recording the most subtle movements. Unless the device’s software is equipped to interpret this data, most of it will be thrown away: it would be too energy intensive to send the raw data to another device for processing.
- In the edge AI world, greenfield projects are ones where the hardware and software are designed together from the ground up. For instance, modern cellphones are designed to include dedicated low-power digital signal processing hardware so that they can continually listen out for a wake word (such as “OK, Google” or “Hey, Siri”) without draining the battery. The hardware is chosen with the specific wake word–detection algorithm in mind
- In contrast, brownfield edge AI projects begin with existing hardware that was originally designed for a different purpose. Developers must work within the constraints of the existing hardware to bring AI capabilities to a product. This reduces developers’ freedom, but it avoids the major costs and risks associated with designing new hardware. For example, a developer could add wake word detection to a Bluetooth audio headset that is already on the market by making use of spare cycles in the device’s existing embedded processor. This new functionality could even be added to existing devices with a firmware update.