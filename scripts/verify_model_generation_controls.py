"""Offline regression checks for model-generation download and safety controls.

These checks do not establish GPU reproduction; that requires two real full runs.
"""
from __future__ import annotations
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fetcher',ROOT/'cloud/fetch_progen_checkpoint.py')
fetcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


class DownloadControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.destination=Path(self.temp.name)/'checkpoint'
        self.bytes=b'pinned-test-asset'
        self.assets={'model.safetensors':(len(self.bytes),hashlib.sha256(self.bytes).hexdigest())}
        self.pin=patch.object(fetcher,'ASSETS',self.assets)
        self.pin.start()

    def tearDown(self):
        self.pin.stop()
        self.temp.cleanup()

    def test_anonymous_download(self):
        with patch.object(fetcher.urllib.request,'urlopen',return_value=io.BytesIO(self.bytes)) as request,patch.object(fetcher.subprocess,'run') as cli:
            fetcher.fetch(self.destination,source='public')
        self.assertEqual((self.destination/'model.safetensors').read_bytes(),self.bytes)
        self.assertTrue(request.call_args.args[0].startswith('https://github.com/RishyanthReddy/amp-challenge-public/releases/download/'))
        cli.assert_not_called()

    def test_matching_cache_needs_no_network(self):
        self.destination.mkdir()
        (self.destination/'model.safetensors').write_bytes(self.bytes)
        with patch.object(fetcher.urllib.request,'urlopen') as request:
            fetcher.fetch(self.destination,source='public')
        request.assert_not_called()

    def test_bad_cached_asset_is_preserved(self):
        self.destination.mkdir()
        (self.destination/'model.safetensors').write_bytes(b'unknown')
        with patch.object(fetcher.urllib.request,'urlopen') as request:
            with self.assertRaises(ValueError):fetcher.fetch(self.destination,source='public')
        self.assertEqual((self.destination/'model.safetensors').read_bytes(),b'unknown')
        request.assert_not_called()

    def test_oversized_download_is_not_installed(self):
        with patch.object(fetcher.urllib.request,'urlopen',return_value=io.BytesIO(self.bytes+b'overflow')):
            with self.assertRaises(ValueError):fetcher.fetch(self.destination,source='public')
        self.assertFalse((self.destination/'model.safetensors').exists())

    def test_hash_mismatch_is_not_installed(self):
        with patch.object(fetcher.urllib.request,'urlopen',return_value=io.BytesIO(b'x'*len(self.bytes))):
            with self.assertRaises(ValueError):fetcher.fetch(self.destination,source='public')
        self.assertFalse((self.destination/'model.safetensors').exists())

    def test_unknown_download_mode_is_rejected(self):
        with self.assertRaises(ValueError):fetcher.fetch(self.destination,source='unknown')


class GenerationControls(unittest.TestCase):
    def test_different_regeneration_preserves_canonical_files(self):
        from amp_challenge_2027 import model_generation as generator
        from amp_challenge_2027 import generate as exporter
        def different_export(top, library, output, reference):
            output.mkdir(parents=True)
            for name in generator.EXPECTED:
                (output/name).write_bytes(b'different regenerated result')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for folder in ['generate','generate_broad_spectrum']:
                (root/folder).mkdir()
                for name in generator.EXPECTED:
                    (root/folder/name).write_bytes(b'original submission')
            with patch.object(generator,'ROOT',root),patch.object(generator.sys,'platform','linux'),patch.object(generator.shutil,'which',return_value='/usr/bin/nvidia-smi'),patch.object(generator,'run'),patch.object(generator,'copy_inputs'),patch.object(exporter,'main',side_effect=different_export):
                with self.assertRaisesRegex(RuntimeError,'Regenerated output differs'):
                    generator.generate()
            for folder in ['generate','generate_broad_spectrum']:
                for name in generator.EXPECTED:
                    self.assertEqual((root/folder/name).read_bytes(),b'original submission')
            self.assertFalse((root/'generation_runs/latest_generation.json').exists())

    def test_root_environment_has_its_own_directory(self):
        from amp_challenge_2027 import model_generation as generator
        with patch.object(generator.subprocess,'run') as call:
            generator.run(ROOT,'.','cloud/fetch_progen_checkpoint.py')
        self.assertEqual(call.call_args.kwargs['env']['UV_PROJECT_ENVIRONMENT'],str(ROOT/'.generation-envs/root'))

    def test_unsupported_host_preserves_outputs(self):
        from amp_challenge_2027 import model_generation as generator
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old=root/'generate/library.fasta'
            old.parent.mkdir();old.write_bytes(b'untouched')
            with patch.object(generator,'ROOT',root),patch.object(generator.sys,'platform','darwin'),patch.object(generator.subprocess,'run') as call:
                with self.assertRaisesRegex(RuntimeError,'RTX 4090'):generator.generate()
            call.assert_not_called()
            self.assertEqual(old.read_bytes(),b'untouched')
            self.assertFalse((root/'generation_runs').exists())

    def test_model_entry_points(self):
        import tomllib
        entry=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['scripts']
        self.assertEqual(entry['generate'],'amp_challenge_2027.model_generation:cli')
        self.assertEqual(entry['generate_broad_spectrum'],'amp_challenge_2027.model_generation:cli')
        self.assertEqual(entry['export_submission'],'amp_challenge_2027.generate:cli')


class PortableOrdering(unittest.TestCase):
    def test_generic_keys_preserve_values_and_dtypes(self):
        import sys
        import pandas as pd
        sys.path.insert(0,str(ROOT/'data-engineering/src'))
        from amp_data.ordering import sort_descending_portable
        frame=pd.DataFrame({'score':[.9,.9,.1,.9,.5], 'label':['a','b','c','d','e']})
        result=sort_descending_portable(frame,'score')
        self.assertEqual(result.dtypes.to_dict(),frame.dtypes.to_dict())
        self.assertEqual(result['score'].tolist(),[.9,.9,.9,.5,.1])
        self.assertEqual(set(result['label']),set(frame['label']))
        self.assertEqual(frame['label'].tolist(),['a','b','c','d','e'])

    def test_invalid_score_is_rejected(self):
        import sys
        import pandas as pd
        sys.path.insert(0,str(ROOT/'data-engineering/src'))
        from amp_data.ordering import sort_descending_portable
        with self.assertRaises(ValueError):sort_descending_portable(pd.DataFrame({'score':[float('nan')]}),'score')


class ApexIsolation(unittest.TestCase):
    def test_apex_does_not_replace_live_uv_environment(self):
        import os
        from types import SimpleNamespace
        import pandas as pd
        spec=importlib.util.spec_from_file_location('apex_wrapper',ROOT/'shared-evaluator/src/evaluator/apex_scorer.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            apex=Path(directory)
            (apex/'APEX_predict.py').touch()
            captured=[]
            def fake_scoring(cmd,**kwargs):
                captured.append(kwargs['env'])
                out=Path(cmd[cmd.index('-o')+1])
                pd.DataFrame({name:[1.0] for name in module.PATHOGEN_PANEL},index=['KKLLKKLL']).to_csv(out)
                return SimpleNamespace(returncode=0,stderr='')
            with patch.dict(os.environ,{'UV_PROJECT_ENVIRONMENT':'caller-environment','VIRTUAL_ENV':'caller-venv'}),patch.object(module.subprocess,'run',side_effect=fake_scoring):
                result=module.ApexScorer(apex).score_batch(['KKLLKKLL'])
            self.assertEqual(result.loc['KKLLKKLL','apex_mean_mic'],1.0)
            self.assertNotIn('UV_PROJECT_ENVIRONMENT',captured[0])
            self.assertNotIn('VIRTUAL_ENV',captured[0])


if __name__=='__main__':
    unittest.main()
