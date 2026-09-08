"""Offline checks for the bounded amendment and exact-request reuse."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase3_revision as revision
import phase3_run as runner
from phase3_api import QuotaLedger
from review_phase3 import file_hash


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = runner.load_settings()
        cls.original, cls.traces, cls.frozen = runner.frozen_inputs(cls.settings)

    def test_only_three_development_questions_change(self):
        amended, changed = revision.amended_cases(self.original)
        self.assertEqual(len(changed), 3)
        for old, new in zip(self.original, amended):
            if old['case_id'] not in changed:
                self.assertEqual(old, new)
                continue
            self.assertEqual(new['partition'], 'development')
            self.assertEqual(new['question'], revision.QUESTION)
            for field in set(old) - {'question', 'revision', 'review'}:
                self.assertEqual(old[field], new[field], field)
            self.assertNotIn('bodies', new['question'])
        counterfactual = next(c for c in amended if c['origin'] == {'file': 'authored_cases.jsonl', 'row': 8})
        self.assertEqual(counterfactual['question'].replace('San Francisco', 'San Diego'), revision.QUESTION)

    def test_missing_source_evidence_refused(self):
        original = copy.deepcopy(self.original)
        next(c for c in original if c['origin'].get('row') == 13)['source_evidence'][0]['text'] = 'No source proof.'
        with self.assertRaises(ValueError):
            revision.amended_cases(original)

    def test_exact_payload_reuse_and_context_change_invalidates(self):
        amended, changed = revision.amended_cases(self.original)
        old = {c['case_id']: c for c in self.original}
        checks = revision.reusable_requests(amended, self.traces, old, self.traces, self.settings)
        self.assertEqual(sum(r['reused'] for r in checks), 394)
        self.assertEqual({r['case_id'] for r in checks if not r['reused']}, set(changed))
        traces = copy.deepcopy(self.traces)
        cid = next(c['case_id'] for c in amended if c['case_id'] not in changed)
        traces[cid]['contexts'][0]['text'] += ' changed'
        checks = revision.reusable_requests(amended, traces, old, self.traces, self.settings)
        self.assertEqual(sum(r['reused'] for r in checks), 392)

    def test_prepare_idempotent_does_not_overwrite_original(self):
        original_settings = self.settings
        paths = [original_settings['work'] / name for name in ('retrievals.json', 'frozen.json')]
        before = [file_hash(p) for p in paths]
        with tempfile.TemporaryDirectory(prefix='revision-test-', dir=original_settings['work']) as directory:
            revised_settings = {**original_settings, 'work': Path(directory), 'reviewed': Path(directory) / 'reviewed'}
            with patch.object(runner, 'load_settings', side_effect=lambda path=None: revised_settings if path else original_settings):
                revision.prepare_revision()
                revision.prepare_revision()
            traces = json.loads((Path(directory) / 'retrievals.json').read_text())['traces']
            self.assertEqual(len(traces), 199)
            control = next(c for c in self.original if c['origin']['file'] == 'review_queue.jsonl' and c['origin']['row'] == 13)
            self.assertNotIn(control['case_id'], traces)
            self.assertEqual(len(runner.read_jsonl(Path(directory) / 'retrieval_review.jsonl')), 155)
        self.assertEqual(before, [file_hash(p) for p in paths])

    def test_seed_preserves_four_failures_and_no_request_rows(self):
        cases, _ = revision.amended_cases(self.original)
        original_settings = self.settings
        results = [json.loads((original_settings['work'] / stage / 'results.json').read_text()) for stage in ('development', 'final')]
        old_run = results[0]['identity']
        with tempfile.TemporaryDirectory() as directory:
            settings = {**original_settings, 'work': Path(directory), 'reviewed': Path(directory) / 'reviewed'}
            ledger = QuotaLedger(Path(directory) / 'test.sqlite')
            for result in results:
                for policy in ('B0', 'B1'):
                    for cid, pred in result['predictions'][policy].items():
                        ledger.save_prediction(old_run, cid, policy, pred)
            for _ in range(2):
                with patch.object(runner, 'load_settings', side_effect=lambda path=None: settings if path else original_settings), \
                     patch.object(runner, 'frozen_inputs', side_effect=[(cases, self.traces, {'test': True}), (self.original, self.traces, self.frozen)]), \
                     patch.object(revision, 'QuotaLedger', return_value=ledger):
                    audit = revision.seed_unchanged_predictions()
            predictions = {p: ledger.predictions_for(audit['revision_run'], p) for p in ('B0', 'B1')}
            self.assertEqual(sum(len(v) for v in predictions.values()), 394)
            self.assertEqual(sum(v['status'] == 'exhausted' for rows in predictions.values() for v in rows.values()), 4)
            self.assertEqual(ledger.usage(audit['revision_run'])['http_attempts'], 0)


if __name__ == '__main__':
    unittest.main()
