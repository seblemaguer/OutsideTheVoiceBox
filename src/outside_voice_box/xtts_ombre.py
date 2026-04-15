#!/usr/bin/env python
# coding: utf-8

import pathlib
import sys
import torch

import torch.nn.functional as F
import numpy as np
from scipy.io.wavfile import write as wave_write

from outside_voice_box.model import setup_tts_model, download_model_by_name, load_config


# function to time align latents and transition from one to the other over time
def blend_latents_numpy(a_np, b_np) -> torch.Tensor:
    # Convert from numpy to torch tensors
    a = torch.from_numpy(a_np).float().unsqueeze(0)  # [1, T_a, 1024]
    b = torch.from_numpy(b_np).float().unsqueeze(0)  # [1, T_b, 1024]

    T_a = a.shape[1]
    T_b = b.shape[1]
    T_m = int((T_a + T_b) / 2)

    # Interpolate to T_m
    a_interp = F.interpolate(a.transpose(1, 2), size=T_m, mode="linear", align_corners=True).transpose(1, 2)
    b_interp = F.interpolate(b.transpose(1, 2), size=T_m, mode="linear", align_corners=True).transpose(1, 2)

    # Create weights from 0 to 1 across T_m and blend
    weights = torch.linspace(0, 1, T_m).view(1, T_m, 1)
    blended = (1 - weights) * a_interp + weights * b_interp  # [1, T_m, 1024]

    return blended


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f"The command should be {sys.argv[0]} <root_model_path> <sample_path>")
        sys.exit(-1)

    model_dir = pathlib.Path(sys.argv[1])
    output_dir = pathlib.Path(sys.argv[2])
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    # Download the required model
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_path, config_path, vocoder_path, vocoder_config_path, _ = download_model_by_name(model_name)

    # load fine tuned model on VCTK corpus, can be skipped for using the published model
    config = load_config(model_dir / "config.json")
    tts_model = setup_tts_model(config)
    tts_model.load_checkpoint(config, checkpoint_path=model_dir / "best_model.pth", checkpoint_dir=model_dir, eval=True)
    if torch.cuda.is_available():
        tts_model.cuda

    # create references (for Ombré) and an equal weighted output (for vocoder input)
    refwav = [model_dir / "ref_samples/p226_023.wav", model_dir / "ref_samples/p262_023.wav"]
    wghts = [[1.7, 1.0], [1.0, 1.7], [0.5, 0.5]]
    audios = []
    for i in range(len(wghts)):
        audio = tts_model.synthesize(
            text="Implicit bias is like the invisible weight of water, pushing down on us all the time, but only noticeable when we try to swim upstream.",
            config=config,
            speaker_id=None,
            voice_dirs=None,
            d_vector=None,
            temperature=0.9,
            # speed=1.1,
            repetition_penalty=20.0,
            speaker_wav=refwav,
            audio_weights=wghts[i],
            language="en",
        )
        audio_out = np.asarray(audio["wav"])
        output_file = output_dir/f"synth_weights_{'_'.join([str(x) for x in wghts[i]])}.wav"
        print(f"Saving audio for weights f{wghts[i]} to {output_file}")
        wave_write(output_file, config.audio.output_sample_rate, audio_out)
        audios.append(audio)

    # Blend in the latent space
    ombre_latent = blend_latents_numpy(audios[0]["gpt_latents"][0], audios[1]["gpt_latents"][0])
    inverse_latent = blend_latents_numpy(audios[1]["gpt_latents"][0], audios[0]["gpt_latents"][0])

    # create ombré audio
    with torch.no_grad():
        wav_ombre = tts_model.hifigan_decoder(ombre_latent, g=audios[2]["speaker_embedding"]).cpu().squeeze()
        wav_ombre = np.asarray(wav_ombre)
        print(f"Saving \"ombre.wav\" audio in {output_dir}" )
        wave_write(output_dir/"ombre.wav", config.audio.output_sample_rate, wav_ombre)
        wav_inverse_ombre = tts_model.hifigan_decoder(inverse_latent, g=audios[2]["speaker_embedding"]).cpu().squeeze()
        wav_inverse_ombre = np.asarray(wav_inverse_ombre)
        print(f"Saving \"inverse_ombre.wav\" audio in {output_dir}" )
        wave_write(output_dir/"inverse_ombre.wav", config.audio.output_sample_rate, wav_inverse_ombre)
