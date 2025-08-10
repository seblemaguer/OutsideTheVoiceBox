# OutsideTheVoiceBox

Codes and samples for "Outside the Voice Box: Rethinking AI Voice Design Beyond Embodied Constraints" (under review) that presents four proposed concepts of disembodied flexibility for Text-to-Speech (TTS) systems:

We generated the audio illustrations for each concept by combining and modifying recent neural TTS methods to achieve the required controllability. The corresponding code and implementation details are shared in this repo.

## Concepts
### Essence
The TTS model trained on spontaneous speech described in [1] used to generate the audio samples which exhibit embodied human-like spontaneous expressions such as laughter, tongue clicks and breathing. An implementation of this with pre-trained models can be found at https://github.com/evaszekely/So_To_Speak
Voice quality variations (breathy voice and vocal fry) were added using the open-source voice conversion tool CreakVC [2] with published implementation at https://github.com/Hfkml/CreakVC

### Expand
To generate speech samples spanning a wide range of perceived vocal tract lengths (influencing the degree of resonance), we used the multi-speaker TTS architecture introduced in [3]. This approach applies constrained PCA to x-vector speaker embeddings, explicitly isolating prosodic components while incorporating the acoustic Vocal Tract Length (aVTL) as a controllable parameter. https://github.com/evaszekely/ambiguous

### Palette
In order to generate the effect of smooth transitioning between speaker identities, we implemented a modification to the XTTS model code (released on https://github.com/coqui-ai/TTS) to allow weighted averaging across multiple speaker embeddings during inference. The modified inference function (in xtts.py) operates on the Conditioning Encoder, which generates a fixed-size embedding used to condition both speaker identity and prosody. To use the illustrative Notebook (https://github.com/evaszekely/OutsideTheVoiceBox/blob/main/XTTS%20modified%20model%20for%20generating%20Palette%20samples.ipynb) the model definition saved in the model/xtts.py function replaces the model under TTS/tts/models in the original repo; this is compatible with the published pre-trained model or can be used with a finetuned model containing the target speakers. For the Palette concept, the weights across speakers sum to 100%.

### Ombré
In order to create a similar effect of speaker transition within a single sentence, we transitioned the GPT encoder latents (serving as input to the XTTS decoder function [4]) between two inputs over the temporal axis. This is demonstrated in the illustrative Notebook (https://github.com/evaszekely/OutsideTheVoiceBox/blob/main/XTTS%20modified%20model%20for%20generating%20Ombré%20samples.ipynb).
Using the modified model described under Palette, we first created two versions of the same utterance, each with the majority of the weight on one speaker. The GPT encoder latents for each version were time-aligned to remove any residual duration mismatch. A new latent sequence was then constructed by linearly interpolating from one latent at the start to the other at the end, which served as input to the model decoder. The decoder was conditioned on a pretrained speaker embedding [4] formed by an equal-weighted average of the two original samples.

## References
[1] Éva Székely, Jeff Higginbotham, and Francesco Possemato. Voice and choice: Investigating the role of prosodic variation in request compliance and perceived politeness using conversational tts. In 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 466–476. Association for Computational Linguistics (ACL), 2024.

[2] Harm Lameris, Joakim Gustafson, and Éva Székely. Creakvc: A voice conversion tool for modulating creaky voice. In 25th Interspeech Conferece 2024, Kos Island, Greece, September 1-5, 2024, pages 1005–1006. International Speech Communication Association, 2024.

[3] Éva Székely and Maxwell Hope. An inclusive approach to creating a palette of synthetic voices for gender diversity. In Proc. Interspeech, pages 3070–3074, 2024.

[4] Edresson Casanova, Kelly Davis, Eren Gölge, Görkem Göknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, and Julian Weber. XTTS: a massively multilingual zero-shot text-to-speech model. In Proc. Interspeech, pages 4978–4982, 2024.

The Github implementations linked here are each governed by their respective licences.
