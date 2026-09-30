from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from . import personas


class PersonaCatalogTests(SimpleTestCase):
    # SimpleTestCase forbids database queries, including accidental session writes.
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        settings_override = override_settings(PERSONACHAT_ROOT=self.root)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        personas._read_catalog.cache_clear()
        self.addCleanup(personas._read_catalog.cache_clear)

    def write(self, suffix, profiles):
        path = self.root / f"personachat_truecased_{suffix}.json"
        path.write_text(json.dumps([{"personality": facts} for facts in profiles]), encoding="utf-8")
        return path

    def test_complete_profiles_are_deduplicated_across_all_dataset_files(self):
        self.write("full_train.part1", [["I like books.", "I have a dog."]])
        self.write("full_train.part2", [["I teach."]])
        self.write("full_train.part3", [["I swim."]])
        self.write("full_valid", [["I have a dog.", " I  like books.\n", "I have a dog."], ["I sing."]])
        self.write("sample_train", [["I like books.", "I have a dog."], ["I paint."]])
        self.write("sample_valid", [["I dance."]])
        (self.root / "unrelated.json").write_text("invalid", encoding="utf-8")

        catalog = personas.get_personas()

        self.assertEqual(len(catalog), 6)
        self.assertEqual(catalog[0]["facts"], ["I like books.", "I have a dog."])
        self.assertEqual(catalog[0]["sources"], ["Train", "Validation"])
        self.assertEqual([entry["facts"] for entry in catalog[1:]], [
            ["I teach."], ["I swim."], ["I sing."], ["I paint."], ["I dance."],
        ])
        self.assertEqual(catalog[-1]["sources"], ["Validation"])
        self.assertEqual(len({entry["id"] for entry in catalog}), 6)

    def test_ids_are_stable_if_files_or_fact_order_change(self):
        path = self.write("full_train.part1", [["I read.", "I swim."]])
        original = personas.get_personas()[0]["id"]
        path.unlink()
        self.write("sample_valid", [[" I  swim. ", "I read."]])
        self.assertEqual(personas.get_personas()[0]["id"], original)

    def test_cache_reuses_parsed_data_and_does_not_expose_mutable_entries(self):
        self.write("full_train.part1", [["I read."]])
        with patch.object(personas.json, "load", wraps=json.load) as load:
            first = personas.get_personas()
            first[0]["facts"].append("Changed")
            first[0]["sources"].clear()
            first.clear()
            second = personas.get_personas()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(second[0]["facts"], ["I read."])
        self.assertEqual(second[0]["sources"], ["Train"])

    def test_cache_invalidates_for_mtime_size_added_and_removed_files(self):
        path = self.write("full_train.part1", [["I read."]])
        original = path.stat()
        self.assertEqual(personas.get_personas()[0]["facts"], ["I read."])

        self.write("full_train.part1", [["I cook."]])
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
        self.assertEqual(path.stat().st_size, original.st_size)
        self.assertEqual(personas.get_personas()[0]["facts"], ["I cook."])

        modified = path.stat()
        self.write("full_train.part1", [["I cook every day."]])
        os.utime(path, ns=(modified.st_atime_ns, modified.st_mtime_ns))
        self.assertEqual(personas.get_personas()[0]["facts"], ["I cook every day."])

        added = self.write("sample_valid", [["I sing."]])
        self.assertEqual(len(personas.get_personas()), 2)
        added.unlink()
        self.assertEqual(len(personas.get_personas()), 1)
        self.assertLessEqual(personas._read_catalog.cache_info().currsize, 2)

    def test_parallel_requests_only_parse_once(self):
        self.write("full_train.part1", [["I read."]])
        with patch.object(personas.json, "load", wraps=json.load) as load:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: personas.get_personas(), range(8)))
        self.assertEqual(load.call_count, 1)
        self.assertTrue(all(result == results[0] for result in results))

    def test_missing_and_empty_directory(self):
        self.assertEqual(personas.get_personas(), [])
        with override_settings(PERSONACHAT_ROOT=self.root / "missing"):
            self.assertEqual(personas.get_personas(), [])

    def test_invalid_json_and_profiles_produce_readable_errors(self):
        path = self.root / "personachat_truecased_full_train.part1.json"
        for invalid in ["{", "{}", "[null]", '[{"personality": "I read."}]',
                        '[{"personality": []}]', '[{"personality": [null]}]',
                        '[{"personality": ["  "]}]']:
            with self.subTest(contents=invalid):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, path.name):
                    personas.get_personas()
        with override_settings(PERSONACHAT_ROOT=path):
            with self.assertRaises(NotADirectoryError):
                personas.get_personas()
