#!/usr/bin/env python
# coding: utf-8

import pathlib
import sys
import torch

import numpy as np
from scipy.io.wavfile import write as wave_write

from TTS.config import load_config

from outside_voice_box.model import setup_tts_model, download_model_by_name, load_config


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

    # create Palette synthesis
    refwav = [model_dir / "ref_samples/p226_023.wav", model_dir / "ref_samples/p262_023.wav"]
    refwav = [
        "models/GPT_XTTS_v2.0_vctk_rfrm/ref_samples/p226_023.wav",
        "models/GPT_XTTS_v2.0_vctk_rfrm/ref_samples/p262_023.wav",
    ]
    wghts = [[1.25, -0.25], [1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0], [-0.25, 1.25]]
    for i in range(len(wghts)):
        audio = tts_model.synthesize(
            text="Listen to the samples in order and see if they change, then return to the first one.",
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
