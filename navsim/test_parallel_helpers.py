"""Check disjoint cache ownership, global normalization and paired aggregation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from parallel_helpers import assigned_logs, merge_cache, merge_evaluation, verify_cache, visible_gpu_identifiers
from protocol import write_json, sha256_file


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        teacher = self.root / 'teacher'
        teacher.write_text('fixture')
        manifest = self.root / 'filters.json'
        write_json(manifest, dict(train=dict(tokens=['a','b']), test=dict(tokens=['a','b','c','d'])))
        self.config = dict(filter_manifest=str(manifest), ltf_checkpoint=str(teacher), identity_tolerance=1e-6,
            gaps=[1,2], arms=['identity','featsim_ltf'], dataset='fixture', tokens_file=None,
            protocol='OpenScene-LTF-OL-v1')

    def tearDown(self):
        self.temp.cleanup()

    def test_log_ownership(self):
        mapping = {str(i): f'log_{i//3}' for i in range(31)}
        shards = assigned_logs(mapping, 8)
        self.assertEqual(set.union(*shards), set(mapping.values()))
        self.assertEqual(sum(map(len, shards)), len(set(mapping.values())))
        self.assertEqual(shards, assigned_logs(dict(reversed(list(mapping.items()))), 8))

    def test_platform_gpu_mask(self):
        self.assertEqual(visible_gpu_identifiers(3, {'CUDA_VISIBLE_DEVICES':'3,7,11'}), ['3','7','11'])
        self.assertEqual(visible_gpu_identifiers(2, {'CUDA_VISIBLE_DEVICES':'GPU-abc,GPU-def'}), ['GPU-abc','GPU-def'])
        with self.assertRaises(ValueError):
            visible_gpu_identifiers(8, {'CUDA_VISIBLE_DEVICES':'0,1'})

    def cache_fixture(self):
        import torch
        for rank, token in enumerate(('a','b')):
            root = self.root / 'cache_shards' / f'rank_{rank:02d}'
            log = f'log_{rank}'
            write_json(root / 'complete.json', dict(complete=True, config=self.config, rank=rank, world=2,
                ltf_sha256=sha256_file(self.config['ltf_checkpoint']), tokens=[token], logs=[log], frames=[token]))
            write_json(root / 'feature_cut_identity.json', dict(max_abs=0., shapes=[[1,1,1,1]]*4))
            write_json(root / 'pairs.json', [dict(token=token, log=log, gap=gap, source=token, target=token) for gap in (1,2)])
            # Worker0: [1,3], worker1: [5,7,9]. Correct mean=5, var=8.
            values = torch.tensor([1.,3.] if rank == 0 else [5.,7.,9.], dtype=torch.float64)
            torch.save(dict(sums=[values.sum().view(1)]*4, squares=[values.square().sum().view(1)]*4,
                counts=[len(values)]*4), root / 'moments.pt')

    def test_global_cache_moments(self):
        import torch
        self.cache_fixture()
        merge_cache(self.config, self.root, 2)
        norm = torch.load(self.root / 'normalization.pt', weights_only=False)
        torch.testing.assert_close(norm['mean'][0], torch.tensor([5.]))
        torch.testing.assert_close(norm['std'][0], torch.tensor([8.]).sqrt())

    def test_reject_overlapping_cache(self):
        self.cache_fixture()
        path = self.root / 'cache_shards/rank_01/complete.json'
        info = json.loads(path.read_text())
        info['frames'] = ['a']
        write_json(path, info)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            merge_cache(self.config, self.root, 2)
        self.assertFalse((self.root / 'CACHE_COMPLETE.json').exists())

    def test_multiworker_cache_handoff_to_single_worker(self):
        self.config.update(preparation_workers=2,world_size=1)
        self.cache_fixture()
        merge_cache(self.config,self.root,2)
        receipt = verify_cache(self.config,self.root)
        self.assertEqual(receipt['workers'],2)
        self.assertEqual(receipt['train_tokens'],2)
        self.assertEqual(receipt['train_pairs'],4)
        # Worker counts are independently configured; optimizer uses one process.
        self.assertEqual(self.config['world_size'],1)

    def test_cache_handoff_rejects_modified_pairs(self):
        self.config.update(preparation_workers=2,world_size=1)
        self.cache_fixture()
        merge_cache(self.config,self.root,2)
        write_json(self.root / 'pairs.json',[])
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            verify_cache(self.config,self.root)

    def test_cache_handoff_rejects_changed_teacher(self):
        self.config.update(preparation_workers=2,world_size=1)
        self.cache_fixture()
        merge_cache(self.config,self.root,2)
        Path(self.config['ltf_checkpoint']).write_text('changed teacher')
        with self.assertRaisesRegex(ValueError,'teacher changed'):
            verify_cache(self.config,self.root)

    def eval_fixture(self):
        from eval_open_loop import finalize_evaluation
        values = dict(a=0., b=0., c=0., d=100.)
        for rank, tokens in enumerate((['a','c'], ['b','d'])):
            root = self.root / 'shards' / f'rank_{rank:02d}'
            records = [dict(token=token, log_name='log_x' if token != 'd' else 'log_y', arm=arm, gap_frames=gap,
                metrics={'planning/ade_agreement':values[token], 'planning/fde_agreement':values[token], 'gt/ade_4s_m':values[token]})
                for token in tokens for arm in self.config['arms'] for gap in self.config['gaps']]
            identities = [dict(token=token, trajectory=0., agent_states=0., agent_labels=0., bev_semantic_map=0.) for token in tokens]
            config = {**self.config, 'tokens_file':str(root / 'requested_tokens.json')}
            write_json(root / 'tokens.json', tokens)
            finalize_evaluation(config, root, records, identities, {'teacher':'same'})

    def test_global_log_mean(self):
        self.eval_fixture()
        merge_evaluation(self.config, self.root, 2)
        summary = json.loads((self.root / 'stats.json').read_text())
        self.assertEqual(summary['num_tokens'], 4)
        self.assertEqual(summary['num_logs'], 2)
        self.assertEqual(summary['table'][0]['per_token_mean']['gt/ade_4s_m'], 25.)
        self.assertEqual(summary['table'][0]['per_log_macro']['gt/ade_4s_m'], 50.)

    def test_reject_missing_pair(self):
        self.eval_fixture()
        path = self.root / 'shards/rank_01/records.json'
        records = json.loads(path.read_text())
        write_json(path, records[:-1])
        with self.assertRaisesRegex(ValueError, 'Incomplete paired set'):
            merge_evaluation(self.config, self.root, 2)
        self.assertFalse((self.root / 'stats.json').exists())

    def test_reject_duplicate_pair(self):
        self.eval_fixture()
        path = self.root / 'shards/rank_01/records.json'
        records = json.loads(path.read_text())
        write_json(path, records + [records[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate record'):
            merge_evaluation(self.config, self.root, 2)

    def test_reject_teacher_mismatch(self):
        self.eval_fixture()
        path = self.root / 'shards/rank_01/stats.json'
        summary = json.loads(path.read_text())
        summary['artifacts']['teacher'] = 'other'
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, 'hashes differ'):
            merge_evaluation(self.config, self.root, 2)


if __name__ == '__main__':
    unittest.main()
