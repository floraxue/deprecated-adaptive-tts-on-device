# Adaptive TTS On-Device
Adaptive speech synthesis from text on-device

# On Device Training
## Our goal
1. Amount of data is small: Pretrained model + few shot learning: the training data in devices is small.
2. Explore more data types: Currently only using images as input.  Support machine translation etc.
3. Moving from algorithm to application: Write a mobile application with on device training.

## Motivations for On-Device Training
1. Customization: Customized model weights for each device
2. Economy: No data transmission overhead. 
3. Privacy: Data does not leave devices

## Constraints of On-Device Training
1. Memory Limit: 
Total Memory = Parameter Memory + Gradient Memory + Layer Activations Memory. 
2. Speed Limit (FLOPS):
Not a hard limit. We don’t need training to be finished in real-time.
3. Energy/Battery Limit:
Not a hard limit. We can limit training to be only executed during charging.

# Adaptive TTS
Given a piece of text and a small piece of a user's voice, generate a waveform as if the user is speaking this text.

## Architecture
Synthesizer (to get mel spectrograms) + Embedding (encodes the user's identity) + Vocoder (generates audio from mel spectrograms.)

## Papers for reference
1. [Wavenet: A Generative Model For Raw Audio](https://arxiv.org/pdf/1609.03499.pdf)
2. [Tacotron 2](https://arxiv.org/pdf/1712.05884.pdf)
3. [Tacotron 2 Implementation](https://github.com/NVIDIA/tacotron2)
4. [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/pdf/1811.00002.pdf)
5. [WaveGlow Implementation](https://github.com/NVIDIA/waveglow)
6. [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://arxiv.org/pdf/1906.00794.pdf)
7. [Sample Efficient Adaptive Text-to-Speech](https://arxiv.org/pdf/1809.10460.pdf)
8. [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/pdf/1905.09263.pdf)
9. [FastSpeech Implementation](https://github.com/xcmyz/FastSpeech)
