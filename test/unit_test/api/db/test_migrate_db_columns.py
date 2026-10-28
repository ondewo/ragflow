"""
Tests that ``migrate_db`` adds the columns the models declare.

``init_database_tables`` creates missing TABLES only, so a column added to a
table that already exists reaches an upgraded database solely through a
``migrate_db`` call. A column that is declared on the model but never migrated
is invisible on a fresh install and breaks every query against an upgraded one,
which is why the expectations here are derived from the model classes instead
of being spelled out a second time.
"""

import inspect

import pytest
from peewee import BooleanField, FloatField, TextField

import api.db.db_models as db_models
from api.db.db_models import Dialog, TenantModelInstance

pytestmark = pytest.mark.p2

TENANT_MODEL_ID_HELP_TEXT = "id in tenant_model"


def _models_by_table() -> dict[str, type]:
    """Every persisted model, keyed by its table name."""
    tables: dict[str, type] = {}
    for _name, model in inspect.getmembers(db_models, inspect.isclass):
        if model is db_models.DataBaseModel or not issubclass(model, db_models.DataBaseModel):
            continue
        tables[model._meta.table_name] = model
    return tables


def _declared_tenant_model_id_columns() -> dict[str, set[str]]:
    """The ``id in tenant_model`` columns the models declare, keyed by table.

    Two independent markers are unioned so that neither a reworded ``help_text``
    nor a column named outside the ``tenant_<kind>_id`` scheme can quietly empty
    the expectation and make the tests below vacuous.
    """
    declared: dict[str, set[str]] = {}
    for table_name, model in _models_by_table().items():
        columns = {
            field.name
            for field in model._meta.fields.values()
            if field.help_text == TENANT_MODEL_ID_HELP_TEXT or (field.name.startswith("tenant_") and field.name.endswith("_id") and field.name != "tenant_id")
        }
        if columns:
            declared[table_name] = columns
    return declared


@pytest.fixture
def migration_calls(monkeypatch) -> tuple[dict[tuple[str, str], object], dict[tuple[str, str], object]]:
    """Run ``migrate_db`` against a recording stand-in for the schema migrator.

    Returns the added columns and the retyped columns, each keyed by
    ``(table, column)`` with the peewee field the migration passes.
    """
    added: dict[tuple[str, str], object] = {}
    retyped: dict[tuple[str, str], object] = {}

    monkeypatch.setattr(db_models, "alter_db_add_column", lambda migrator, table_name, column_name, column_type: added.setdefault((table_name, column_name), column_type))
    monkeypatch.setattr(db_models, "alter_db_column_type", lambda migrator, table_name, column_name, new_column_type: retyped.setdefault((table_name, column_name), new_column_type))
    monkeypatch.setattr(db_models, "alter_db_rename_column", lambda *args, **kwargs: None)
    monkeypatch.setattr(db_models, "alter_db_drop_index", lambda *args, **kwargs: None)
    # Everything below talks to a live database; migrate_db only has to reach it.
    monkeypatch.setattr(db_models, "migrate", lambda *args, **kwargs: None)
    monkeypatch.setattr(db_models, "relax_gaussdb_empty_string_compatible_columns", lambda: None)
    monkeypatch.setattr(db_models, "migrate_add_unique_email", lambda migrator: None)
    monkeypatch.setattr(db_models, "migrate_model_type_names", lambda: None)
    monkeypatch.setattr(db_models, "ensure_model_indexes", lambda migrator: None)
    monkeypatch.setattr(db_models, "migrate_tenant_model_id_column_types", lambda migrator: None)
    monkeypatch.setattr(db_models, "migrate_postgres_family_model_provider_tables", lambda: None)
    monkeypatch.setattr(db_models, "migrate_document_meta_fields", lambda: None)

    db_models.migrate_db()

    return added, retyped


class TestTenantModelIdColumns:
    """The columns through which the pre-existing tables reach ``tenant_model``."""

    def test_the_column_list_matches_what_the_models_declare(self):
        declared = _declared_tenant_model_id_columns()
        listed: dict[str, set[str]] = {}
        for table_name, column_name in db_models.TENANT_MODEL_ID_COLUMNS:
            listed.setdefault(table_name, set()).add(column_name)

        assert declared, "no tenant_model id columns found - the markers no longer match the models"
        assert listed == declared

    def test_a_missing_column_is_added_as_the_model_declares_it(self, monkeypatch):
        added: dict[tuple[str, str], object] = {}
        monkeypatch.setattr(db_models, "is_gaussdb_compatible_database", lambda: False)
        monkeypatch.setattr(db_models.DB, "table_exists", lambda table_name: True)
        monkeypatch.setattr(db_models, "_get_column_data_type", lambda table_name, column_name: None)
        monkeypatch.setattr(db_models, "alter_db_add_column", lambda migrator, table_name, column_name, column_type: added.setdefault((table_name, column_name), column_type))

        db_models.migrate_tenant_model_id_column_types(migrator=None)

        models_by_table = _models_by_table()
        for table_name, column_names in _declared_tenant_model_id_columns().items():
            for column_name in column_names:
                assert (table_name, column_name) in added, f"{table_name}.{column_name} is never added to an upgraded database"

                declared_field = models_by_table[table_name]._meta.fields[column_name]
                migrated_field = added[(table_name, column_name)]
                assert type(migrated_field) is type(declared_field)
                assert migrated_field.max_length == declared_field.max_length
                assert migrated_field.null == declared_field.null
                # peewee's add_column only emits CREATE INDEX when the field asks for one, so a
                # migration that dropped index=True would leave the column present but unindexed.
                assert migrated_field.index == declared_field.index
        # Only the declared columns, never fragments of their names.
        assert set(added) == {(table_name, column_name) for table_name, column_names in _declared_tenant_model_id_columns().items() for column_name in column_names}


class TestDialogDedupColumns:
    """The retrieval dedup knobs are per-chat settings, so they are Dialog columns."""

    def test_the_columns_are_declared_with_the_documented_defaults(self):
        assert type(Dialog.dedup_threshold) is FloatField
        assert Dialog.dedup_threshold.default == 0.0
        assert type(Dialog.dedup_before_rerank) is BooleanField
        assert Dialog.dedup_before_rerank.default is False

    @pytest.mark.parametrize("column_name", ["dedup_threshold", "dedup_before_rerank"])
    def test_migrate_db_adds_the_column(self, migration_calls, column_name):
        added, _retyped = migration_calls

        assert ("dialog", column_name) in added

        declared_field = Dialog._meta.fields[column_name]
        migrated_field = added[("dialog", column_name)]
        assert type(migrated_field) is type(declared_field)
        assert migrated_field.default == declared_field.default
        assert migrated_field.null == declared_field.null


class TestTenantModelInstanceExtra:
    """``extra`` holds the per-model ``default_headers`` dict, so it needs the room."""

    def test_the_column_is_declared_as_text(self):
        assert type(TenantModelInstance.extra) is TextField

    def test_migrate_db_widens_the_column(self, migration_calls):
        _added, retyped = migration_calls

        assert ("tenant_model_instance", "extra") in retyped

        declared_field = TenantModelInstance._meta.fields["extra"]
        migrated_field = retyped[("tenant_model_instance", "extra")]
        assert type(migrated_field) is type(declared_field)
        assert migrated_field.default == declared_field.default
