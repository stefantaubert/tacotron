import unittest

from tacotron.core.accent_symbols import (get_accent_symbol_id,
                                          get_accent_symbols_count,
                                          get_symbol_id)


class UnitTests(unittest.TestCase):

  # region get_accent_symbols_count()

  def test_get_accent_symbols_count_acc_shared1_ns10_na5_returns_46(self):
    res = get_accent_symbols_count(
      n_symbols=10,
      n_accents=5,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    # 1 + 5 accents * 9 symbols
    self.assertEqual(46, res)

  def test_get_accent_symbols_count_acc_shared2_ns10_na5_returns_42(self):
    res = get_accent_symbols_count(
      n_symbols=10,
      n_accents=5,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    # 2 + 5 accents * 8 symbols
    self.assertEqual(42, res)

  def test_get_accent_symbols_count_acc_shared1_ns1_na50_returns_1(self):
    res = get_accent_symbols_count(
      n_symbols=1,
      n_accents=50,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(1, res)

  def test_get_accent_symbols_count_acc_shared2_ns2_na50_returns_1(self):
    res = get_accent_symbols_count(
      n_symbols=2,
      n_accents=50,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(2, res)

  def test_get_accent_symbols_count_ns10_na5_returns_10(self):
    res = get_accent_symbols_count(
      n_symbols=10,
      n_accents=5,
      accents_use_own_symbols=False,
      shared_symbol_count=1
    )

    self.assertEqual(10, res)
  # endregion

  # region get_accent_symbol_id()

  def test_get_accent_symbol_id_acc_shared0_s0_not_always_returns_0(self):
    res = 0
    for accent_id in range(10):
      res += get_accent_symbol_id(
        symbol_id=0,
        accent_id=accent_id,
        n_symbols=10,
        accents_use_own_symbols=True,
        shared_symbol_count=0
      )

    self.assertNotEqual(0 * 10, res)

  def test_get_accent_symbol_id_acc_shared1_s0_always_returns_0(self):
    res = 0
    for accent_id in range(10):
      res += get_accent_symbol_id(
        symbol_id=0,
        accent_id=accent_id,
        n_symbols=10,
        accents_use_own_symbols=True,
        shared_symbol_count=1
      )

    self.assertEqual(0 * 10, res)

  def test_get_accent_symbol_id_acc_shared2_s1_always_returns_1(self):
    res = 0
    for accent_id in range(10):
      res += get_accent_symbol_id(
        symbol_id=1,
        accent_id=accent_id,
        n_symbols=10,
        accents_use_own_symbols=True,
        shared_symbol_count=2
      )

    self.assertEqual(1 * 10, res)

  def test_get_accent_symbol_id_acc_shared1_s1_a1_ns2_returns_2(self):
    res = get_accent_symbol_id(
      symbol_id=1,
      accent_id=1,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(2, res)

  def test_get_accent_symbol_id_acc_shared1_s1_a1_ns3_returns_2(self):
    res = get_accent_symbol_id(
      symbol_id=1,
      accent_id=1,
      n_symbols=3,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(3, res)

  def test_get_accent_symbol_id_acc_shared1_s1_a3_ns2_returns_4(self):
    res = get_accent_symbol_id(
      symbol_id=1,
      accent_id=3,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(4, res)

  def test_get_accent_symbol_id_acc_shared0_s5_a0_ns10_returns_5(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=0,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(5, res)

  def test_get_accent_symbol_id_acc_shared1_s5_a0_ns10_returns_5(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=0,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(5, res)

  def test_get_accent_symbol_id_acc_shared2_s5_a0_ns10_returns_5(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=0,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(5, res)

  def test_get_accent_symbol_id_acc_shared0_s5_a1_ns10_returns_15(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=1,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(15, res)

  def test_get_accent_symbol_id_acc_shared1_s5_a1_ns10_returns_14(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=1,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(14, res)

  def test_get_accent_symbol_id_acc_shared2_s5_a1_ns10_returns_13(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=1,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(13, res)

  def test_get_accent_symbol_id_acc_shared0_s0_a1_ns6_returns_6(self):
    res = get_accent_symbol_id(
      symbol_id=0,
      accent_id=1,
      n_symbols=6,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(6, res)

  def test_get_accent_symbol_id_acc_shared1_s1_a1_ns6_returns_6(self):
    res = get_accent_symbol_id(
      symbol_id=1,
      accent_id=1,
      n_symbols=6,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(6, res)

  def test_get_accent_symbol_id_acc_shared2_s2_a1_ns6_returns_6(self):
    res = get_accent_symbol_id(
      symbol_id=2,
      accent_id=1,
      n_symbols=6,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(6, res)

  def test_get_accent_symbol_id_acc_shared0_s5_a2_ns10_returns_25(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=2,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(25, res)

  def test_get_accent_symbol_id_acc_shared1_s5_a2_ns10_returns_23(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=2,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(23, res)

  def test_get_accent_symbol_id_acc_shared2_s5_a2_ns10_returns_21(self):
    res = get_accent_symbol_id(
      symbol_id=5,
      accent_id=2,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(21, res)
  # endregion

  # region get_symbol_id()
  def test_get_symbol_id_acc_shared0_m0_ns2_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=0,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(0, res)

  def test_get_symbol_id_acc_shared1_m0_ns2_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=0,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(0, res)

  def test_get_symbol_id_acc_shared0_m2_ns2_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=2,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(0, res)

  def test_get_symbol_id_acc_shared1_m2_ns2_returns_1(self):
    res = get_symbol_id(
      accent_symbol_id=2,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(1, res)

  def test_get_symbol_id_acc_shared2_m3_ns3_returns_2(self):
    res = get_symbol_id(
      accent_symbol_id=3,
      n_symbols=3,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(2, res)

  def test_get_symbol_id_acc_shared1_m3_ns3_returns_1(self):
    res = get_symbol_id(
      accent_symbol_id=3,  # is accent 1
      n_symbols=3,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(1, res)

  def test_get_symbol_id_acc_shared1_m4_ns2_returns_1(self):
    res = get_symbol_id(
      accent_symbol_id=4,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(1, res)

  def test_get_symbol_id_acc_shared0_m4_ns2_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=4,
      n_symbols=2,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(0, res)

  def test_get_symbol_id_acc_shared0_m6_ns6_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=6,
      n_symbols=6,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(0, res)

  def test_get_symbol_id_acc_shared1_m6_ns6_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=6,
      n_symbols=6,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(1, res)

  def test_get_symbol_id_acc_shared2_m6_ns6_returns_0(self):
    res = get_symbol_id(
      accent_symbol_id=6,
      n_symbols=6,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(2, res)

  def test_get_symbol_id_acc_shared0_m15_ns10_returns_5(self):
    res = get_symbol_id(
      accent_symbol_id=15,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=0
    )

    self.assertEqual(5, res)

  def test_get_symbol_id_acc_shared1_m14_ns10_returns_5(self):
    res = get_symbol_id(
      accent_symbol_id=14,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=1
    )

    self.assertEqual(5, res)

  def test_get_symbol_id_acc_shared2_m13_ns10_returns_5(self):
    res = get_symbol_id(
      accent_symbol_id=13,
      n_symbols=10,
      accents_use_own_symbols=True,
      shared_symbol_count=2
    )

    self.assertEqual(5, res)

  # endregion


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
