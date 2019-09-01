# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import json
import os
from copy import deepcopy

import numpy as np
import torch
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from denoiser import Denoiser
from mel2samp import Mel2Samp


def create_reverse_dict(inp):
    reverse = {}
    for k, v in inp.items():
        assert v not in reverse
        reverse[v] = k
    return reverse


def save_audio_chunks(frames, filename, stride, sr=22050, ymax=0.98,
                      normalize=True):
    # Generate stream
    y = torch.zeros((len(frames) - 1) * stride + len(frames[0]))
    for i, x in enumerate(frames):
        y[i * stride:i * stride + len(x)] += x
    # To numpy & deemph
    y = y.numpy().astype(np.float32)
    # if deemph>0:
    #     y=deemphasis(y,alpha=deemph)
    # Normalize
    if normalize:
        y -= np.mean(y)
        mx = np.max(np.abs(y))
        if mx > 0:
            y *= ymax / mx
    else:
        y = np.clip(y, -ymax, ymax)
    # To 16 bit & save
    write(filename, sr, np.array(y * 32767, dtype=np.int16))
    return y


def main(waveglow_path, sigma, output_dir, is_fp16, denoiser_strength):
    # mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    testset = Mel2Samp(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    # train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    test_loader = DataLoader(testset, num_workers=0, shuffle=False,
                              # sampler=train_sampler,
                              batch_size=12,
                              pin_memory=False,
                              drop_last=True)

    speakers_to_sids = deepcopy(testset.speakers)
    sids_to_speakers = create_reverse_dict(speakers_to_sids)
    ut_to_uids = deepcopy(testset.utterances)
    uids_to_ut = create_reverse_dict(ut_to_uids)

    sid_target = np.random.randint(len(speakers_to_sids))
    speaker_target = sids_to_speakers[sid_target]
    sid_target = torch.LongTensor([[sid_target] *
                                   test_loader.batch_size]).view(
        test_loader.batch_size, 1).to(device)

    audios = []
    mels = []
    n_audios = 0
    for i, batch in enumerate(test_loader):
        mel_source, _, sid_source, uid_source, is_last = batch
        mel_source = mel_source.to(device)
        import pdb
        pdb.set_trace()

        with torch.no_grad():
            predicted = waveglow.infer(mel_source, sigma=sigma)
            if denoiser_strength > 0:
                predicted = denoiser(predicted, denoiser_strength)
            # predicted = predicted * MAX_WAV_VALUE

        for j in range(len(predicted)):
            p = predicted[j].cpu()
            audios.append(p)
            mels.append(mel_source[j].cpu())
            speaker_source = sids_to_speakers[sid_source[j].data.item()]
            ut_source = uids_to_ut[uid_source[j].data.item()]
            last = is_last[j].data.item()
            if last:
                ## Hacking to print mel_source here
                fname = os.path.join(output_dir,
                                     "{}_{}_mel.pt".format(speaker_source, ut_source))
                pdb.set_trace()
                torch.save(mels, fname)
                print("Saved mel to {}".format(fname))
                ##

                audio_path = os.path.join(
                    output_dir,
                    "{}_{}_to_{}_synthesis.wav".format(speaker_source,
                                                       ut_source,
                                                       speaker_target))
                print("Synthesizing file No.{} at {}".format(n_audios,
                                                             audio_path))
                save_audio_chunks(audios, audio_path, data_config['stride'],
                                  data_config['sampling_rate'])

                audios = []
                mels = []
                n_audios += 1


def test_pretrained_wg_infer(waveglow_path, mel_path, output_dir, sigma,
                             ymax=0.98):
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    # concat the mels
    import pdb
    pdb.set_trace()
    mel = torch.load(mel_path)
    mel = torch.cat(mel, dim=1)
    mel = torch.autograd.Variable(mel.cuda())
    mel = torch.unsqueeze(mel, 0)

    with torch.no_grad():
        predicted = waveglow.infer(mel, sigma=sigma)

    # post processing
    predicted = predicted.squeeze()
    predicted = predicted.cpu().numpy()
    predicted = predicted.astype(np.float32)
    predicted -= np.mean(predicted)
    mx = np.max(np.abs(predicted))
    if mx > 0:
        predicted *= ymax / mx

    # saving the output
    file_name = os.path.splitext(os.path.basename(mel_path))[0]
    audio_path = os.path.join(
        output_dir, "{}_synthesis.wav".format(file_name))
    write(audio_path, data_config['sampling_rate'],
          np.array(predicted * 32767, dtype=np.int16))
    print(audio_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-m', "--mel_path",
                        help='Path to the mel for testing on pretrained')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    # parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    global data_config
    data_config = config["data_config"]
    data_config['split'] = 'test'
    global device
    device = 'cuda'

    # test_pretrained_wg_infer(args.waveglow_path, args.mel_path,
    #                          args.output_dir, args.sigma)

    main(args.waveglow_path, args.sigma, args.output_dir,
         args.is_fp16, args.denoiser_strength)

