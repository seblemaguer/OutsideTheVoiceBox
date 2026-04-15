import librosa
import torch
import torch.nn.functional as F
from coqpit import Coqpit

from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import split_sentence
from TTS.tts.models.xtts import load_audio, Xtts

init_stream_support()


class ModifiedXtts(Xtts):
    """Overriden XTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config)

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
        audio_weights=None,  # DIFF: added to the original Xtts
    ):
        """Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 24000.
        """
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt (DIFF: modified from the original Xtts)
        if audio_weights is None:
            full_audio = torch.cat(audios, dim=-1)
            gpt_cond_latents = self.get_gpt_cond_latents(
                full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
            )  # [1, 1024, T]
        else:
            comb_embs = []
            num_audio = min(len(audios), len(audio_weights))
            for j in range(num_audio):
                ind_emb = self.get_gpt_cond_latents(
                    audios[j], load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
                )
                comb_embs.append(ind_emb)
            weighted_embs = sum(w * t for w, t in zip(audio_weights[:num_audio], comb_embs))
            gpt_cond_latents = weighted_embs / sum(audio_weights[:num_audio])

        if speaker_embeddings:  # DIFF: modified from the original Xtts
            if audio_weights is None:
                speaker_embedding = torch.stack(speaker_embeddings)
                speaker_embedding = speaker_embedding.mean(dim=0)
            else:
                num_emb = min(len(speaker_embeddings), len(audio_weights))
                speaker_embs = sum(w * t for w, t in zip(audio_weights[:num_emb], speaker_embeddings))
                speaker_embedding = speaker_embs / sum(audio_weights[:num_emb])

        return gpt_cond_latents, speaker_embedding

    def synthesize(
        self, text, config, speaker_wav, language, speaker_id=None, audio_weights=None, **kwargs
    ):  # DIFF: modified from the original Xtts - added the audio_weights parameter
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (XttsConfig): Config with inference parameters.
            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
            return self.inference(text, language, gpt_cond_latent, speaker_embedding, **settings)
        settings.update(
            {
                "gpt_cond_len": config.gpt_cond_len,
                "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
                "max_ref_len": config.max_ref_len,
                "sound_norm_refs": config.sound_norm_refs,
            }
        )
        return self.full_inference(text, speaker_wav, language, audio_weights, **settings) # DIFF: modified from the original Xtts - added the audio_weights parameter

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        audio_weights=None,  # DIFF: added to the original Xtts
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Cloning
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3
                seconds long.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used
                else the first `gpt_cond_len` secs is used. Defaults to 30 seconds.

            gpt_cond_chunk_len: (int) Chunk length used for cloning. It must be <= `gpt_cond_len`.
                If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to 6 seconds.

            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
            audio_weights=audio_weights,  # DIFF: added to the original Xtts
        )

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):  # pylint: disable=unused-argument
        return ModifiedXtts(config)
