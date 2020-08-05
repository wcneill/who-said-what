import unittest
from wave_encoder import *

class TestEncoder(unittest.TestCase):
    def test_plot_no_fit1(self):
        encoder = WaveEncoder()
        self.assertRaises(TypeError, encoder.plot_signal)

    def test_plot_no_fit2(self):
        encoder = WaveEncoder()
        self.assertRaises(ValueError, encoder.plot_transform)

    def test_plot_no_fit3(self):
        encoder = WaveEncoder()
        self.assertRaises(TypeError, encoder.plot_components)

