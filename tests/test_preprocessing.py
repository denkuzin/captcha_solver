import preprocessing
import unittest
import numpy as np


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.raw_small_image = np.random.uniform(0, 255, (16,17,3)).astype(int)
        self.char2ind = {'a': 0,
                         'b': 1,
                         'c': 2}
        self.ind2char = dict((b,a) for a,b in self.char2ind.items())


    def test_randomString(self):
        result = preprocessing.randomString(
            'aaaaaaaaaa', lenght=5)
        self.assertEqual(result, 'aaaaa')

    def test_resize_one(self):
        result = preprocessing.resize_one(
            self.raw_small_image, shape=(32, 32, 3))

        self.assertEqual(result.shape, (32, 32, 3))

    def test_OHE(self):
        result = preprocessing.OHE('abc', self.char2ind)
        correct_result = np.array([[1,0,0],[0,1,0],[0,0,1]])
        cond = np.array_equal(result, correct_result)
        self.assertTrue(cond)
