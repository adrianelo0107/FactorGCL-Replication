"""Train only the archived full-FactorGCL A-share baseline with seed 42."""
import argparse
import hashlib
import json

from factorgcl.config import load_config, resolve_path
from factorgcl.data import prepare_cache
from factorgcl import training as engine

_SIGNATURE = engine.experiment_signature
_EVALUATE = engine.evaluate_dataset


def signature(cfg, fold, cache):
    """Retain the exact historical experiment-signature namespace."""
    spec = json.dumps(cfg.notes['baseline_protocol'], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256((_SIGNATURE(cfg, fold, cache) + '|baseline-v1|' + spec).encode()).hexdigest()


def install_validation_log(cfg, fold_name, resume):
    directory = resolve_path(cfg.training.output_dir) / fold_name
    directory.mkdir(parents=True, exist_ok=True)
    start = engine.load_checkpoint(directory / 'last.pt')['epoch'] if resume and (directory / 'last.pt').exists() else 0
    sidecar = directory / 'validation_epochs.jsonl'
    if sidecar.exists():
        records = [json.loads(line) for line in sidecar.read_text(encoding='utf-8').splitlines() if line.strip()]
        sidecar.write_text(''.join(json.dumps(row) + '\n' for row in records if row['epoch'] <= start), encoding='utf-8')
    epoch = [start]

    def evaluate(*args, **kwargs):
        result = _EVALUATE(*args, **kwargs)
        output_dir = args[4] if len(args) > 4 else kwargs.get('output_dir')
        if output_dir is None:
            epoch[0] += 1
            row = {'epoch': epoch[0], 'validation': result[0], 'training_label_clipping': None}
            with sidecar.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return result

    engine.experiment_signature = signature
    engine.evaluate_dataset = evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='code/configs/ashare_seed42.json')
    parser.add_argument('--fold', choices=['fold_1', 'fold_2'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg.training.device = args.device
    spec = cfg.notes.get('baseline_protocol', {})
    if cfg.training.seed != 42 or spec.get('id') != 'seed_42' or spec.get('training_label_clip_quantiles') is not None:
        raise ValueError('This entry point is restricted to the unclipped-label seed-42 baseline.')
    if cfg.training.scaler_reference_dir is not None or not all((cfg.model.use_prior, cfg.model.use_hidden, cfg.model.use_alpha, cfg.model.use_cl)):
        raise ValueError('Use all FactorGCL modules and fit each fold scaler from its training data.')
    root = resolve_path(cfg.training.output_dir).resolve()
    archive = resolve_path('results').resolve()
    if root == archive or archive in root.parents:
        raise ValueError('Training output must not overwrite the published results/ archive.')
    cache = prepare_cache(cfg.data)
    if args.prepare_only:
        print(json.dumps(cache.manifest, indent=2))
        return
    try:
        for fold in cfg.folds:
            if args.fold and fold.name != args.fold:
                continue
            install_validation_log(cfg, fold.name, args.resume)
            print(engine.run_experiment(cfg, args.resume, fold.name), flush=True)
    finally:
        engine.experiment_signature = _SIGNATURE
        engine.evaluate_dataset = _EVALUATE


if __name__ == '__main__':
    main()
