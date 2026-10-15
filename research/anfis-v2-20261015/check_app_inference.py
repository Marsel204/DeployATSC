"""Compare the existing app inference engine with frozen native predictions.

Run using the app's Python environment. This creates no observation session and
never starts camera acquisition or hardware actuation.
"""
import json
from pathlib import Path
import sys

import numpy as np

from verify_bundle import verify_bundle

ROOT = Path(__file__).resolve().parent


def main():
    verify_bundle()
    sys.path.insert(0, str(ROOT.parents[1]/'jetson-atsc-core'))
    from src.control.anfis_inference import ANFISInferenceEngine
    engine = ANFISInferenceEngine(str(ROOT/'model/candidate_anfis.onnx'), prefer_cuda=False)
    reference = np.load(ROOT/'evidence/app/native_reference.npz', allow_pickle=False)
    inputs, expected = reference['inputs'], reference['predictions']
    actual = engine.session.run([engine.output_name], {engine.input_name: inputs})[0]
    difference = float(np.max(abs(actual-expected)))
    if not np.isfinite(actual).all() or difference > .001:
        raise ValueError(f'ONNX/native parity failed: {difference}')
    for row, native in zip(inputs, expected):
        green = engine.evaluate_green_time(*map(float, row))
        if engine.last_error is not None or not 10 <= green <= 120:
            raise ValueError(f'App inference failed: {engine.last_error}')
        if abs(green-float(native[0])) > .011:
            raise ValueError('Rounded app output does not match native reference')
    print(json.dumps(dict(status='passed', inputs=len(inputs),
                         max_onnx_native_difference_sec=difference,
                         provider=engine.active_provider, fallback_count=0), indent=2))


if __name__ == '__main__':
    main()
