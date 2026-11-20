"""Cached frames/conditions are never recomputed or overwritten on restart."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
import train_full


class CacheResumeTests(unittest.TestCase):
    def run_fixture(self, missing=False):
        import torch
        import ltf_features
        import scene_layout
        from protocol import write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            cache.mkdir()
            teacher = root / 'teacher.stub'
            teacher.write_text('fixture')
            write_json(root / 'filters.json',dict(train=dict(tokens=['a','b'])))
            channels = (2,4,6,8)
            stages = [torch.full((1,c,1,1),float(level+1)) for level,c in enumerate(channels)]
            frames = {}
            for token in ('a','b'):
                keys = [token+'0',token+'1',token+'2',token]
                frames[token] = [dict(token=key,log_name='log0') for key in keys]
                for key in keys:
                    torch.save([t[0].clone() for t in stages],cache / (key+'.pt'))
                for gap in (1,2,3):
                    torch.save(dict(layout=torch.zeros(6,1,1),action=torch.zeros(4)),cache / f'{token}_{gap}.pt')
            if missing:
                (cache / 'a0.pt').unlink()
            before = {p.name:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns) for p in cache.iterdir()}
            scene = Mock()
            builder = Mock()
            builder.compute_features_with_scene_gap.return_value = dict(
                camera_feature=torch.zeros(1,1,1),camera_feature_past=torch.zeros(1,1,1))
            loader = Mock(tokens_stage_one=['a','b'],scene_frames_dicts=frames)
            loader.get_scene_from_token.return_value = scene
            agent = Mock()
            agent.get_feature_builders.return_value = [builder]
            config = dict(device='cpu',gaps=[1,2,3],identity_tolerance=1e-6,
                filter_manifest=str(root / 'filters.json'),ltf_checkpoint=str(teacher),preparation_workers=1)
            with patch.object(train_full,'initialize',return_value=(loader,agent)), \
                    patch.object(ltf_features,'extract_camera_stages',return_value=(stages,torch.zeros(1))) as extract, \
                    patch.object(ltf_features,'replay_camera_stages',return_value=torch.zeros(1)), \
                    patch.object(scene_layout,'condition_layout',side_effect=AssertionError('cached condition recomputed')), \
                    patch.object(scene_layout,'action_between',side_effect=AssertionError('cached action recomputed')):
                train_full.prepare(config,root)
            # Fully cached input loads one scene solely for the final identity
            # self-check; missing input loads one scene solely for that frame.
            self.assertEqual(loader.get_scene_from_token.call_count,1)
            self.assertEqual(builder.compute_features_with_scene_gap.call_count,1)
            self.assertEqual(extract.call_count,1)
            for name,receipt in before.items():
                path = cache / name
                self.assertEqual((hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_mtime_ns),receipt)
            progress = json.loads((root / 'cache_shards/rank_00/cache_progress.json').read_text())
            self.assertEqual(progress['extracted_frames'],int(missing))
            self.assertEqual(progress['reused_frames'],8-int(missing))
            self.assertEqual(progress['reused_conditions'],6)
            self.assertEqual(progress['new_conditions'],0)
            self.assertTrue((root / 'CACHE_COMPLETE.json').exists())
            normalization = torch.load(root / 'normalization.pt',weights_only=False)
            for level,c in enumerate(channels):
                torch.testing.assert_close(normalization['mean'][level],torch.full((c,),float(level+1)))

    def test_existing_cache_skips_camera_work_and_preserves_files(self):
        self.run_fixture()

    def test_only_missing_frame_is_extracted(self):
        self.run_fixture(missing=True)


if __name__ == '__main__':
    unittest.main()
