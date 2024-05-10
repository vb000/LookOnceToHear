import os, glob
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr
from resemblyzer import VoiceEncoder, preprocess_wav

import src.utils as utils

def load_model(config, run_dir, device):
    checkpoint = os.path.join(run_dir, 'best.ckpt')

    # Load model and state dict
    model = utils.import_attr(config.pl_module)(**config.pl_module_args)
    if os.path.exists(checkpoint):
        print(f"Loading {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Warn if checkpoint doesn't exist
        print(f'Warning: no checkpoint found in {run_dir}')

    model.eval()
    model.to(device)

    return model

def dvector_from_tensor(encoder, x, sr):
    embeddings = []
    for i in range(x.shape[0]):
        audio = x[i].mean(0).numpy()
        audio = preprocess_wav(audio, sr)
        embed = torch.from_numpy(encoder.embed_utterance(audio))
        embeddings.append(embed)
    return torch.stack(embeddings)

def speech_durations(x, sr):
    durantions = []
    for e in x:
        e, _ = librosa.effects.trim(e, top_db=30)
        durantions.append(librosa.get_duration(y=e, sr=sr))
    return durantions

def _sample_to_df(inputs, targets):
    results = []
    for i in range(inputs['mixture'].shape[0]):
        tgt_idx = inputs['tgt_idx'][i]
        src_files = [os.path.basename(_[i]) for _ in inputs['source_files']][1:]
        src_genders = [_[1][i] for _ in inputs['speaker_info']]
        src_embeds = [_[i] for _ in inputs['mixture_embeddings']]
        src_sisnr = inputs['mixture_sisnr'][i].item()
        src_embed_dist = []
        for j in range(len(src_embeds)):
            if j != tgt_idx:
                src_embed_dist.append(F.cosine_similarity(
                    src_embeds[tgt_idx], src_embeds[j], dim=-1).item())
        src_embed_dist = max(src_embed_dist)
        enroll_src_files = [os.path.basename(_[i]) for _ in inputs['enrollments_source_files']][1:]
        enroll_sisnr = inputs['enrollments_sisnr'][i].item()
        enroll_embed_gt = targets['embedding_gt'][i]
        enroll_embed_dist = max([
            F.cosine_similarity(enroll_embed_gt, targets['embedding_neg'][0][i], dim=-1).item(),
            F.cosine_similarity(enroll_embed_gt, targets['embedding_neg'][1][i], dim=-1).item()])
        tgt_enroll_error = inputs['tgt_enroll_error'][i].item()
        tgt_ang_vel = inputs['tgt_ang_vel'][i].item()
        results.append({
            's1': src_files[0],
            's1g': src_genders[0],
            's2': src_files[1],
            's2g': src_genders[1],
            's3': src_files[2],
            's3g': src_genders[2],
            'input_sisnr': src_sisnr,
            'input_embed_dist': src_embed_dist,
            'e1': enroll_src_files[0],
            'e2': enroll_src_files[1],
            'e3': enroll_src_files[2],
            'enroll_sisnr': enroll_sisnr,
            'enroll_embed_dist': enroll_embed_dist,
            'tgt_enroll_error': tgt_enroll_error,
            'tgt_ang_vel': tgt_ang_vel
        })
    return pd.DataFrame(results)

def run(args, device):
    enroll_cfg_name = os.path.basename(args.enroll_run_dir) \
        if args.enroll_run_dir is not None else 'clean'
    results_file = os.path.join(
        args.run_dir, f'results_{args.dset}_{enroll_cfg_name}.csv')
    assert not os.path.exists(results_file), f'{results_file} already exists'

    # Load config
    config = utils.Params(args.config)
    model = load_model(config, args.run_dir, device)

    # Load enrollment model
    enroll_model = None
    if args.enroll_config is not None:
        enroll_config = utils.Params(args.enroll_config)
        enroll_model = load_model(enroll_config, args.enroll_run_dir, device)

    if args.embed_from_wav:
        encoder = VoiceEncoder()

    # Load dataset
    if args.dset == 'test':
        ds = utils.import_attr(config.test_dataset)(**config.test_data_args)
    elif args.dset == 'val':
        ds = utils.import_attr(config.val_dataset)(**config.val_data_args)
    else:
        raise ValueError(f'Invalid dataset: {args.dset}')
    dl_iter = iter(torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=16))

    results = []
    for i, (inputs, targets) in enumerate(tqdm(dl_iter)):
        with torch.no_grad():
            mixture = inputs['mixture'].to(device)
            if enroll_model is not None and args.embed_from_wav:
                enrollments = inputs['enrollments'].squeeze(1).to(device)
                enrollments = enroll_model.model(enrollments).to('cpu')
                embedding = dvector_from_tensor(encoder, enrollments, ds.sr).to(device)
                embedding = embedding.unsqueeze(1)
            elif enroll_model is not None:
                enrollments = inputs['enrollments'].squeeze(1).to(device)
                embedding = enroll_model.model(enrollments)
                embedding = embedding.unsqueeze(1)
            else:
                embedding = targets['embedding_gt'].to(device)
            outputs = model(mixture, embedding)
            outputs = outputs.cpu()
            embedding_sim = F.cosine_similarity(
                embedding.cpu().squeeze(1), targets['embedding_gt'].squeeze(1), dim=-1)

        # Compute metrics
        output_sisnr = si_snr(outputs, targets['target']).mean(dim=1)
        si_snr_i = si_snr(outputs, targets['target']) - si_snr(inputs['mixture'], targets['target'])
        si_snr_i = si_snr_i.view(si_snr_i.shape[0], -1).mean(dim=1)

        sample_df = _sample_to_df(inputs, targets)
        sample_df['output_sisnr'] = output_sisnr.numpy()
        sample_df['si_snr_i'] = si_snr_i.numpy()
        sample_df['embedding_sim'] = embedding_sim.numpy()
        sample_df['enroll_duration'] = speech_durations(
            inputs['enrollments_clean'].squeeze(1).numpy(), ds.sr)

        results.append(sample_df)

        if args.sample and i == 3:
            print(pd.concat(results, ignore_index=True))
            print("Avergae SI-SNRi:", sample_df['si_snr_i'].mean())
            return

    results = pd.concat(results, ignore_index=True)
    print("Average SI-SNRi:", results['si_snr_i'].mean())
    print("Avergae cosine similarity:", results['embedding_sim'].mean())
    print(f"Writing results to {results_file}")
    results.to_csv(results_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--dset', type=str, default='test')
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else 'cpu'

    config = {
        'config': 'configs/tsh.json',
        'run_dir': 'runs/tsh',
        'enroll_config': 'configs/embed.json',
        'enroll_run_dir': 'runs/embed'
    }

    args.config = config['config']
    args.run_dir = config['run_dir']
    args.enroll_config = config.get('enroll_config', None)
    args.enroll_run_dir = config.get('enroll_run_dir', None)
    args.embed_from_wav = config.get('embed_from_wav', False)

    run(args, device)
