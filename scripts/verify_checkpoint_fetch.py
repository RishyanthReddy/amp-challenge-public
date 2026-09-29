"""Exercise checkpoint cache/download failure handling without network or real weights."""
import hashlib
import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'cloud/fetch_progen_checkpoint.py'
spec = importlib.util.spec_from_file_location('checkpoint_fetch', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
asset = b'known test checkpoint'
pins = {'model.safetensors': (len(asset), hashlib.sha256(asset).hexdigest())}
with tempfile.TemporaryDirectory() as temp, patch.object(module, 'ASSETS', pins):
    root = Path(temp)
    def download(command, **kwargs):
        assert command[:3] == ['gh', 'release', 'download']
        (Path(command[command.index('--dir') + 1]) / 'model.safetensors').write_bytes(asset)
    with patch.object(module.subprocess, 'run', side_effect=download) as call:
        module.fetch(root / 'valid')
        assert call.call_count == 1
        module.fetch(root / 'valid')
        assert call.call_count == 1, 'A valid cache must not contact the network'
    (root / 'valid/model.safetensors').write_bytes(b'corrupt local')
    with patch.object(module.subprocess, 'run') as call:
        try:
            module.fetch(root / 'valid')
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt cache accepted')
        call.assert_not_called()
    def corrupt_download(command, **kwargs):
        (Path(command[command.index('--dir') + 1]) / 'model.safetensors').write_bytes(b'corrupt')
    with patch.object(module.subprocess, 'run', side_effect=corrupt_download):
        try:
            module.fetch(root / 'bad_download')
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt download accepted')
    assert not (root / 'bad_download/model.safetensors').exists()
print('PASS: verified download, cached reuse, corrupt-cache refusal, corrupt-download refusal')
