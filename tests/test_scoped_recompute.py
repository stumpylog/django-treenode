from django.test import TestCase

from tests.models import Category
from treenode.signals import no_signals


class ScopedRecomputeTestCase(TestCase):
    """
    Regression tests for scoped-tree-recompute-design.md.
    update_tree() must scope its read+recompute to the affected tree when
    it safely can, and every scoped result must exactly match what a full
    (unscoped) recompute would produce -- verified by diffing against one,
    not by hand-picking which fields to check.
    """

    def _assert_matches_full_recompute(self):
        _, dirty_instances, _ = Category._TreeNodeModel__get_nodes_data()
        self.assertEqual(
            dirty_instances,
            [],
            f"scoped write drifted from a full recompute on: "
            f"{[d.pk for d in dirty_instances]}",
        )

    def _build_forest(self, num_trees=3, nodes_per_tree=5):
        roots = []
        with no_signals():
            for t in range(num_trees):
                root = Category.objects.create(name=f"tree{t:02d}-root")
                roots.append(root)
                for i in range(nodes_per_tree - 1):
                    Category.objects.create(
                        name=f"tree{t:02d}-node{i:02d}", tn_parent=root
                    )
        Category.update_tree()
        return roots

    def test_insert_leaf_does_not_touch_unrelated_tree_and_matches_full_recompute(self):
        roots = self._build_forest()
        target_root = roots[0]

        before = {
            o.pk: o.tn_order
            for o in Category.objects.exclude(
                tn_parent__in=[target_root] + [target_root.pk]
            )
        }
        Category.objects.create(name="new-leaf", tn_parent=target_root)

        after = {
            o.pk: o.tn_order
            for o in Category.objects.exclude(name="new-leaf").exclude(
                pk__in=[c.pk for c in target_root.get_descendants()]
            )
        }
        # every row outside target_root's tree must be byte-for-byte unchanged
        for pk, order in before.items():
            if pk in after:
                self.assertEqual(after[pk], order)

        self._assert_matches_full_recompute()

    def test_delete_leaf_matches_full_recompute(self):
        self._build_forest()
        victim = Category.objects.get(name="tree00-node00")
        victim.delete()
        self._assert_matches_full_recompute()

    def test_root_siblings_are_not_corrupted_by_a_scoped_write(self):
        """
        Pins the first bug found during design review: a scoped run only
        loads one tree, so a naive implementation sees an incomplete
        sibling group for that tree's root and overwrites its real
        tn_siblings_pks/tn_siblings_count with an empty/wrong one.
        """
        roots = self._build_forest(num_trees=4)
        target_root = roots[0]
        other_roots_before = {
            r.pk: (r.tn_siblings_pks, r.tn_siblings_count, r.tn_index)
            for r in Category.objects.filter(tn_ancestors_count=0)
        }

        Category.objects.create(name="new-leaf", tn_parent=target_root)

        other_roots_after = {
            r.pk: (r.tn_siblings_pks, r.tn_siblings_count, r.tn_index)
            for r in Category.objects.filter(tn_ancestors_count=0)
        }
        self.assertEqual(other_roots_before, other_roots_after)
        self._assert_matches_full_recompute()

    def test_delete_without_cascade_reparents_children_correctly(self):
        """
        Pins the second bug found during design review: delete(cascade=False)
        re-parents children to root level -- a root-level structural change
        that must fall back to a full recompute, not the scoped path.
        """
        a = Category.objects.create(name="a")
        aa = Category.objects.create(name="aa", tn_parent=a)
        aaa = Category.objects.create(name="aaa", tn_parent=aa)
        Category.objects.create(name="b")

        aa.delete(cascade=False)
        aaa.refresh_from_db()

        self.assertIsNone(aaa.tn_parent_id)
        self.assertTrue(aaa.is_root())
        self._assert_matches_full_recompute()

    def test_loaddata_falls_back_to_full_recompute(self):
        """
        Pins the suspected loaddata staleness risk: post_save's raw=True
        must not attempt scoping (a not-yet-finalized fixture row's stored
        ancestor data can't be trusted to compute a scope from).

        Uses CategoryFixtures (not Category) and the existing fixture file
        tests/fixtures/test_fixtures_issue_0088.json, matching
        tests/test_fixtures.py::test_loaddata_issue_0088 -- this is the
        real, only fixture-loading test already in this codebase.
        """
        from django.core.management import call_command

        from tests.models import CategoryFixtures

        call_command("loaddata", "test_fixtures_issue_0088.json")

        _, dirty_instances, _ = CategoryFixtures._TreeNodeModel__get_nodes_data()
        self.assertEqual(
            dirty_instances,
            [],
            f"scoped write drifted from a full recompute on: "
            f"{[d.pk for d in dirty_instances]}",
        )

    def test_reparent_within_tree_falls_back_and_matches_full_recompute(self):
        a = Category.objects.create(name="a")
        b = Category.objects.create(name="b", tn_parent=a)
        c = Category.objects.create(name="c", tn_parent=a)

        b.tn_parent = c
        b.save()

        self._assert_matches_full_recompute()

    def test_reparent_across_trees_falls_back_and_matches_full_recompute(self):
        roots = self._build_forest()
        node = Category.objects.get(name="tree00-node00")
        node.tn_parent = roots[1]
        node.save()

        self._assert_matches_full_recompute()

    def test_new_root_falls_back_and_matches_full_recompute(self):
        self._build_forest()
        Category.objects.create(name="a-new-root")
        self._assert_matches_full_recompute()

    def test_insert_query_count_does_not_grow_with_unrelated_tree_count(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        def insert_and_count_queries(num_trees):
            roots = self._build_forest(num_trees=num_trees, nodes_per_tree=10)
            with CaptureQueriesContext(connection) as ctx:
                Category.objects.create(name="probe-leaf", tn_parent=roots[0])
            Category.objects.all().delete()
            return len(ctx.captured_queries)

        small_forest_queries = insert_and_count_queries(num_trees=3)
        large_forest_queries = insert_and_count_queries(num_trees=30)

        self.assertEqual(small_forest_queries, large_forest_queries)
