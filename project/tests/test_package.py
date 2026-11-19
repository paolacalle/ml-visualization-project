"""Public package imports, caller dotenv lookup, and bundled SQL contracts."""

from contextlib import chdir
import importlib
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cmv_data
from cmv_data import Database, TextStore


class PackageTests(unittest.TestCase):
    def test_public_classes_and_all_modules_import_from_installed_namespace(self):
        self.assertIs(Database, importlib.import_module("cmv_data.SupabaseConnection").Database)
        self.assertIs(TextStore, importlib.import_module("cmv_data.TextStore").TextStore)
        for module in pkgutil.walk_packages(cmv_data.__path__, prefix="cmv_data."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_import_outside_checkout_does_not_connect(self):
        code = """
from unittest.mock import patch
with patch('psycopg.connect', side_effect=AssertionError('connection on import')):
    from cmv_data import Database, TextStore
    assert Database.__module__ == 'cmv_data.SupabaseConnection'
    assert TextStore.__module__ == 'cmv_data.TextStore'
"""
        with tempfile.TemporaryDirectory() as outside:
            subprocess.run([sys.executable, "-I", "-c", code], cwd=outside,
                           capture_output=True, text=True, check=True)


class PackagedDatabaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env_file = self.root / ".env"
        self.password = "raw@${literal}/#%!"
        self.env_file.write_text(
            "DB_HOST=fixture.invalid\nDB_PORT=5432\nDB_NAME=fixture\n"
            f"DB_USER=analyst\nDB_PASSWORD='{self.password}'\n"
        )
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.connection = MagicMock()
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.cursor.description = [("value",)]
        self.cursor.fetchall.return_value = [(1,)]
        self.connect = self.enterContext(patch("cmv_data.SupabaseConnection.psycopg.connect",
                                              return_value=self.connection))

    def test_default_env_is_found_from_callers_directory_with_literal_password(self):
        nested = self.root / "notebooks"
        nested.mkdir()
        with chdir(nested):
            db = Database()
        self.addCleanup(db.close)
        self.connect.assert_called_once_with(host="fixture.invalid", port="5432", dbname="fixture",
                                             user="analyst", password=self.password, sslmode="require")

    def test_explicit_env_file_works_outside_checkout_and_environment_wins(self):
        os.environ["DB_USER"] = "environment_user"
        db = Database(env_file=self.env_file)
        self.addCleanup(db.close)
        self.assertEqual(self.connect.call_args.kwargs["user"], "environment_user")
        self.assertEqual(self.connect.call_args.kwargs["password"], self.password)

    def test_packaged_query_file_resolves_outside_checkout(self):
        db = Database(env_file=self.env_file)
        self.addCleanup(db.close)
        with chdir(self.root):
            self.assertEqual(db.execute("test.sql"), [(1,)])
        self.cursor.execute.assert_called_once_with("SELECT 1;", None)

    def test_sql_parameters_remain_bound_and_close_is_idempotent(self):
        db = Database(env_file=self.env_file)
        statement = "SELECT post_id FROM cmv.posts WHERE post_id = %s;"
        params = ("t3_test",)
        self.assertEqual(db.execute(statement, params), [(1,)])
        self.cursor.execute.assert_called_once_with(statement, params)
        db.close()
        db.close()
        self.connection.close.assert_called_once()
        self.assertIsNone(db.connection)


if __name__ == "__main__":
    unittest.main()
