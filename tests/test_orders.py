"""
Unit tests for orders.py: completions of partial orders.
"""
import unittest
from typing import Iterable

from LearningPriority.orders import PartialOrder, completions_of_poset


def is_total_order(poset: PartialOrder) -> bool:
    elements = list(poset.elements)
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            if (a, b) not in poset.relations and (b, a) not in poset.relations:
                return False
    return True


def extends_relations(base: PartialOrder, completion: PartialOrder) -> bool:
    return base.relations.issubset(completion.relations)


def to_list(iterable: Iterable[PartialOrder]):
    return list(iterable)


class TestCompletionsOfPoset(unittest.TestCase):
    def test_chain_has_single_completion(self):
        elements = {"a", "b", "c"}
        relations = {("a", "b"), ("b", "c")}
        poset = PartialOrder(elements, relations)

        completions = to_list(completions_of_poset(poset))
        self.assertEqual(len(completions), 1)
        self.assertTrue(is_total_order(completions[0]))
        self.assertTrue(extends_relations(poset, completions[0]))

    def test_two_incomparable_elements(self):
        elements = {"a", "b"}
        relations = set()
        poset = PartialOrder(elements, relations)

        completions = to_list(completions_of_poset(poset))
        self.assertEqual(len(completions), 2)
        self.assertTrue(all(is_total_order(c) for c in completions))
        self.assertTrue(all(extends_relations(poset, c) for c in completions))

        has_a_before_b = any(("a", "b") in c.relations for c in completions)
        has_b_before_a = any(("b", "a") in c.relations for c in completions)
        self.assertTrue(has_a_before_b)
        self.assertTrue(has_b_before_a)

    def test_fork_order(self):
        elements = {"a", "b", "c"}
        relations = {("a", "b"), ("a", "c")}
        poset = PartialOrder(elements, relations)

        completions = to_list(completions_of_poset(poset))
        self.assertEqual(len(completions), 2)
        self.assertTrue(all(is_total_order(c) for c in completions))
        self.assertTrue(all(extends_relations(poset, c) for c in completions))

        has_b_before_c = any(("b", "c") in c.relations for c in completions)
        has_c_before_b = any(("c", "b") in c.relations for c in completions)
        self.assertTrue(has_b_before_c)
        self.assertTrue(has_c_before_b)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
