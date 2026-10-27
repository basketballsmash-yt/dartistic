"""Camera skeleton capture and personalized dart outcome learning."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import statistics
import sys
import time
import urllib.request

FEATURES = ['elbow_min_deg', 'elbow_max_deg', 'elbow_range_deg',
            'wrist_peak_body_lengths_s', 'elbow_drift_body_lengths', 'duration_s']


def player_id(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,39}', value):
        raise ValueError('Player ID must be 1-40 lowercase letters, digits, underscores or hyphens.')
    return value


def require_player(data, expected):
    if data.get('player') != expected:
        raise ValueError('Recording and model belong to different players, or player identity is missing.')


def assign_player(args):
    player_id(args.player)
    data, _ = load_throw(args.recording)
    if data.get('player') not in (None, args.player):
        raise ValueError('Recording already belongs to another player; refusing reassignment.')
    data['player'] = args.player
    path = Path(args.recording)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)
    print('Assigned recording to player:', args.player)


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def summarize(frames, hand):
    if hand not in ('left', 'right'):
        raise ValueError('Throwing hand must be left or right.')
    for f in frames:
        if not math.isfinite(f['t']):
            raise ValueError('Invalid timestamp.')
        if f.get('world') is not None:
            if len(f['world']) != 33 or len(f['visibility']) != 33:
                raise ValueError('Expected 33 pose landmarks.')
            if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in f['world']):
                raise ValueError('Invalid landmark coordinates.')
    ids = [12, 14, 16, 24] if hand == 'right' else [11, 13, 15, 23]
    valid = [f for f in frames if f.get('world') and
             all(f['visibility'][i] >= .65 for i in ids)]
    if len(valid) < 12 or len(valid) < .8 * len(frames):
        raise ValueError('Insufficient tracking: keep shoulder, elbow, wrist and hip visible.')
    times = [f['t'] for f in valid]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Frame timestamps must increase.')
    duration = times[-1]-times[0]
    if any(b-a > .15 for a,b in zip(times,times[1:])):
        raise ValueError('Tracking gaps exceed 150 ms; record again with better visibility/frame rate.')
    if duration < .3:
        raise ValueError('Record at least 0.3 seconds of motion.')
    angles, wrists, elbows, scales = [], [], [], []
    for f in valid:
        s, e, w, h = [f['world'][i] for i in ids]
        scale = distance(s, h)
        if scale < .05:
            raise ValueError('Invalid torso scale.')
        a, b = [s[i]-e[i] for i in range(3)], [w[i]-e[i] for i in range(3)]
        denom = distance(s, e)*distance(w, e)
        if denom < 1e-6:
            raise ValueError('Degenerate arm landmarks.')
        angles.append(math.degrees(math.acos(max(-1, min(1, sum(x*y for x,y in zip(a,b))/denom)))))
        wrists.append([w[i]-s[i] for i in range(3)])
        elbows.append([e[i]-s[i] for i in range(3)])
        scales.append(scale)
    scale = statistics.median(scales)
    speeds = [distance(wrists[i], wrists[i-1])/scale/(times[i]-times[i-1])
              for i in range(1, len(valid)) if times[i]-times[i-1] <= .15]
    if not speeds:
        raise ValueError('Frame rate too low for motion analysis.')
    values = [min(angles), max(angles), max(angles)-min(angles), max(speeds),
              max(distance(e, elbows[0]) for e in elbows)/scale, duration]
    return dict(zip(FEATURES, values))


def capture(args):
    player_id(args.player)
    if not args.output:
        args.output = str(Path('profiles') / args.player / 'recordings' /
                          (time.strftime('%Y%m%d-%H%M%S') + '.json'))
    import cv2
    import mediapipe as mp
    model = Path(args.pose_model)
    if not model.exists():
        model.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
        print('Downloading Google pose model...')
        temporary = model.with_suffix('.download')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(model)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise ValueError('Cannot open camera/video. Check camera permissions and index.')
    frames, recording, start, last, index = [], bool(args.video), time.monotonic(), -1, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video and (not math.isfinite(fps) or fps <= 0):
        cap.release()
        raise ValueError('Video has no valid frame rate.')
    edges = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
             (23,24),(23,25),(25,27),(24,26),(26,28)]
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = index/fps if args.video else time.monotonic()-start
                stamp = max(last+1, round(t*1000)); last = stamp; index += 1
                result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), stamp)
                row = {'t': stamp/1000, 'world': None, 'visibility': []}
                if result.pose_world_landmarks:
                    row['world'] = [[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]]
                    row['visibility'] = [p.visibility for p in result.pose_landmarks[0]]
                    points = result.pose_landmarks[0]
                    height, width = frame.shape[:2]
                    for a,b in edges:
                        if min(points[a].visibility, points[b].visibility) >= .65:
                            cv2.line(frame, (int(points[a].x*width),int(points[a].y*height)),
                                     (int(points[b].x*width),int(points[b].y*height)), (80,240,80), 3)
                if recording:
                    frames.append(row)
                cv2.putText(frame, ('RECORDING' if recording else 'READY') + ' | SPACE start/stop | Q finish',
                            (12,30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,220,255), 2)
                cv2.imshow('Dartistic', frame)
                key = cv2.waitKey(1) & 255
                if key == ord('q'):
                    break
                if key == 32:
                    if recording:
                        break
                    recording = True
    finally:
        cap.release(); cv2.destroyAllWindows()
    if not frames:
        raise ValueError('No recording. Press SPACE before throwing.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump({'schema':1, 'source':'mediapipe', 'player':args.player,
                   'hand':args.hand, 'frames':frames}, f)
    print('Saved', output)
    print(json.dumps(summarize(frames, args.hand), indent=2))


def load_throw(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 1:
        raise ValueError('Unsupported recording schema.')
    return data, summarize(data['frames'], data['hand'])


def make_model():
    local = os.environ.get('AI4ANIMATION_PATH', r'C:\Users\shuji\OneDrive\Desktop\dartistic\ai4animationpy')
    if Path(local).is_dir():
        sys.path.insert(0, local)
    from ai4animation.AI.Models.MultiLayerPerceptron import Model
    return Model(len(FEATURES), 1, 32, dropout=0.0)


def outcome_label(hit, target, board='unspecified', miss_distance=None):
    if hit not in ('hit', 'miss') or not target.strip():
        raise ValueError('Specify hit/miss and a target region.')
    if board not in ('steel', 'soft', 'unspecified'):
        raise ValueError('Unknown board type.')
    if miss_distance is not None:
        if hit == 'hit' or board == 'unspecified':
            raise ValueError('Miss distance requires a miss and an explicit board type.')
        if not math.isfinite(miss_distance) or miss_distance <= 0:
            raise ValueError('Miss distance must be positive and finite.')
        if board == 'soft' and not float(miss_distance).is_integer():
            raise ValueError('Soft-tip bit counts must be whole numbers.')
    return {'hit': hit == 'hit', 'target': target.strip().upper(), 'board': board,
            'miss_distance': miss_distance,
            'distance_unit': {'steel':'cm', 'soft':'bits'}.get(board)}


def label(args):
    data, _ = load_throw(args.recording)
    data['outcome'] = outcome_label(args.result, args.target, args.board, args.miss_distance)
    # A region label must not retain an obsolete center-distance training label.
    data.pop('target_error_mm', None)
    path = Path(args.recording)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)
    print(json.dumps(data['outcome'], indent=2))


def training_value(data, objective):
    outcome = data.get('outcome')
    if outcome is None:
        return None
    if objective == 'hit':
        return float(outcome['hit'])
    unit = 'cm' if objective == 'distance-cm' else 'bits'
    if outcome['distance_unit'] != unit:
        return None
    # Zero means inside the target region, never distance to its center.
    return 0.0 if outcome['hit'] else outcome['miss_distance']


def train(args):
    player_id(args.player)
    folder = args.folder or str(Path('profiles') / args.player / 'recordings')
    model_path = args.model or str(Path('profiles') / args.player / 'models' / (args.objective + '.pt'))
    import numpy as np
    import torch
    torch.manual_seed(7)
    rows, labels, hands, contexts = [], [], set(), set()
    skipped = 0
    for path in sorted(Path(folder).glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('excluded') or data.get('quality',{}).get('accepted') is False:
            skipped += 1
            continue
        if data.get('player') != args.player:
            skipped += 1
            continue
        value = training_value(data, args.objective)
        if value is None:
            continue
        data, summary = load_throw(path)
        rows.append([summary[k] for k in FEATURES]); labels.append([value]); hands.add(data['hand'])
        contexts.add((data['outcome']['target'], data['outcome']['board']))
    print(f'Player {args.player}: {len(rows)} usable labeled throws; {skipped} excluded/poor-quality/other-player recordings skipped.')
    if len(rows) < 20 or len(hands) != 1:
        raise ValueError('Need at least 20 labeled throws, all using the same throwing hand.')
    if len(contexts) != 1:
        raise ValueError('Use recordings with one target and one board type per model.')
    x, y = np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Target errors must be finite and nonnegative.')
    # Last 20% are held out chronologically (use timestamp filenames).
    split = int(len(x)*.8)
    if y[:split].std() < 1e-6:
        raise ValueError('Training portion needs varied outcomes (both hits and misses for hit prediction).')
    if args.objective == 'hit' and len(set(y[split:].flatten())) < 2:
        raise ValueError('Held-out portion needs both hits and misses to evaluate hit prediction.')
    mean, std = x[:split].mean(0), x[:split].std(0).clip(.001)
    ym, ys = float(y[:split].mean()), max(float(y[:split].std()), 1.)
    if args.objective == 'hit':
        ym, ys = 0.0, 1.0
    xt = torch.tensor((x-mean)/std); yt = torch.tensor((y-ym)/ys)
    model = make_model(); optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    model.train()
    for _ in range(400):
        optimizer.zero_grad()
        logits = model(xt[:split])
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[:split])
                if args.objective == 'hit' else ((logits-yt[:split])**2).mean())
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        raw = model(xt[split:])
        pred = raw.sigmoid().numpy() if args.objective == 'hit' else (raw.numpy()*ys+ym).clip(0)
    if args.objective == 'hit':
        metric = 'Brier score (lower is better)'
        mae = float(((pred-y[split:])**2).mean())
        baseline = float(((float(y[:split].mean())-y[split:])**2).mean())
    else:
        metric = 'MAE ' + ('cm' if args.objective == 'distance-cm' else 'bits')
        mae = float(np.abs(pred-y[split:]).mean())
        baseline = float(np.abs(ym-y[split:]).mean())
    destination = Path(model_path); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = torch.load(destination, map_location='cpu', weights_only=True)
        if existing.get('player') != args.player:
            raise ValueError('Refusing to overwrite a model belonging to another or unassigned player.')
    torch.save({'state':model.state_dict(), 'features':FEATURES, 'mean':mean.tolist(),
                'std':std.tolist(), 'ym':ym, 'ys':ys, 'hand':next(iter(hands)),
                'objective':args.objective, 'player':args.player, 'training_count':split,
                'context':list(next(iter(contexts))),
                'metric':metric, 'validation':mae, 'baseline':baseline}, destination)
    print(f'Saved {destination}. Held-out {metric}: {mae:.3f}; baseline {baseline:.3f}.')
    if mae >= baseline:
        print('Model does not beat the baseline; collect more varied throws before trusting predictions.')


def analyze(args):
    data, summary = load_throw(args.recording)
    print(json.dumps(summary, indent=2))
    if args.model:
        import torch
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
        if not checkpoint.get('player'):
            raise ValueError('Legacy model has no player identity. Retrain using --player.')
        require_player(data, checkpoint['player'])
        print('Personal model for:', checkpoint['player'])
        if checkpoint['features'] != FEATURES or checkpoint['hand'] != data['hand']:
            raise ValueError('Model feature schema or throwing hand differs.')
        if 'context' in checkpoint and data.get('outcome'):
            context = [data['outcome']['target'], data['outcome']['board']]
            if context != checkpoint['context']:
                raise ValueError('Recording target/board does not match the model.')
        model = make_model(); model.load_state_dict(checkpoint['state']); model.eval()
        x = torch.tensor([[summary[k] for k in FEATURES]])
        with torch.no_grad():
            prediction = model((x-torch.tensor(checkpoint['mean']))/torch.tensor(checkpoint['std'])).item()
        objective = checkpoint.get('objective', 'legacy-mm')
        if objective == 'hit':
            print(f'Estimated target-hit probability: {torch.sigmoid(torch.tensor(prediction)).item():.1%} (not calibrated)')
        else:
            unit = {'distance-cm':'cm', 'distance-bits':'bits', 'legacy-mm':'mm'}[objective]
            print(f"Estimated distance: {max(0,prediction*checkpoint['ys']+checkpoint['ym']):.2f} {unit}")
        if 'context' in checkpoint:
            print('Model target / board:', ' / '.join(checkpoint['context']))
            print(f"Validation {checkpoint['metric']}: {checkpoint['validation']:.3f}; baseline: {checkpoint['baseline']:.3f}")
        else:
            print(f"Validation MAE: {checkpoint['mae_mm']:.1f} mm; baseline: {checkpoint['baseline_mm']:.1f} mm")
        print('Experimental prediction, not a measured dart landing or a causal coaching recommendation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('capture'); p.add_argument('--camera',type=int,default=0)
    p.add_argument('--video'); p.add_argument('--hand',choices=['left','right'],default='right')
    p.add_argument('--pose-model',default='models/pose_landmarker.task'); p.add_argument('--output')
    p.add_argument('--player',required=True)
    p.set_defaults(run=capture)
    p = commands.add_parser('analyze'); p.add_argument('recording'); p.add_argument('--model'); p.set_defaults(run=analyze)
    p = commands.add_parser('train'); p.add_argument('folder',nargs='?'); p.add_argument('--model')
    p.add_argument('--player',required=True)
    p.add_argument('--objective', choices=['hit','distance-cm','distance-bits'], default='hit'); p.set_defaults(run=train)
    p = commands.add_parser('label'); p.add_argument('recording'); p.add_argument('result',choices=['hit','miss'])
    p.add_argument('--target',required=True); p.add_argument('--board',choices=['steel','soft','unspecified'],default='unspecified')
    p.add_argument('--miss-distance',type=float)
    p.set_defaults(run=label)
    p = commands.add_parser('assign-player'); p.add_argument('recording'); p.add_argument('--player',required=True)
    p.set_defaults(run=assign_player)
    p = commands.add_parser('replay'); p.add_argument('recording')
    p.add_argument('--smoke-seconds',type=float,default=0,help=argparse.SUPPRESS)
    def replay(args):
        from replay import launch
        launch(args)
    p.set_defaults(run=replay)
    p = commands.add_parser('practice'); p.add_argument('--player',required=True)
    p.add_argument('--hand',choices=['right','left'],default='right')
    p.add_argument('--board',choices=['steel','soft'],required=True); p.add_argument('--target',required=True)
    p.add_argument('--camera',type=int,default=0); p.add_argument('--pose-model',default='models/pose_landmarker.task')
    def practice(args):
        from practice import launch
        launch(args)
    p.set_defaults(run=practice)
    p = commands.add_parser('compare'); p.add_argument('--player',required=True)
    p.add_argument('--smoke-seconds',type=float,default=0,help=argparse.SUPPRESS)
    def compare(args):
        from compare import launch
        launch(args)
    p.set_defaults(run=compare)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
