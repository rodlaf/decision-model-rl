import unittest
import numpy as np
from exploration_objectives import EpisodicObservationNovelty,discounted_advantages,adapt_entropy

class ExplorationTests(unittest.TestCase):
    def test_past_rewards_do_not_credit_later_actions(self):
        a=discounted_advantages([[1,0,0],[0,0,0]],0.9)
        self.assertGreater(a[0,0],0)
        np.testing.assert_array_equal(a[:,1:],0)
    def test_future_reward_decays(self):
        a=discounted_advantages([[0,0,1],[0,0,0]],0.9)
        self.assertAlmostEqual(a[0,0]/a[0,1],0.9)
        self.assertAlmostEqual(a[0,1]/a[0,2],0.9)
    def test_equal_trajectories_and_finite_values(self):
        np.testing.assert_array_equal(discounted_advantages([[0,1],[0,1]]),0)
    def test_entropy_controller_direction_and_bounds(self):
        self.assertGreater(adapt_entropy(.03,.2,1),.03)
        self.assertLess(adapt_entropy(.03,1.5,1),.03)
        self.assertEqual(adapt_entropy(.2,0,1),.2)
    def test_generic_first_visit_quantization_and_cap(self):
        n=EpisodicObservationNovelty(np.array([0.0,0.0]),bonus=.01,cap=.025,quantum=.1)
        self.assertEqual(n.observe(np.array([0.02,0.0])),0)
        self.assertEqual(n.observe(np.array([.2,0.0])),.01)
        self.assertEqual(n.observe(np.array([.2,0.0])),0)
        self.assertEqual(n.observe(np.array([.4,0.0])),.01)
        self.assertAlmostEqual(n.observe(np.array([.6,0.0])),.005)
        self.assertEqual(n.observe(np.array([.8,0.0])),0)
    def test_generic_observation_structures(self):
        initial={'b':np.array([1.,2.]),'a':(np.array([0.]),3.)}
        n=EpisodicObservationNovelty(initial)
        self.assertEqual(n.observe({'a':(np.array([0.]),3.),'b':np.array([1.,2.])}),0)
    def test_reward_renews_local_novelty_but_not_episode_cap(self):
        n=EpisodicObservationNovelty(np.array([0.]),bonus=.01,cap=.025,segment_cap=.01)
        self.assertEqual(n.observe(np.array([1.])),.01)
        self.assertEqual(n.observe(np.array([2.])),0)
        self.assertEqual(n.observe(np.array([2.]),extrinsic_reward=.1),0)
        self.assertEqual(n.observe(np.array([1.])),.01)
        self.assertAlmostEqual(n.observe(np.array([3.]),extrinsic_reward=.2),0)
        self.assertAlmostEqual(n.observe(np.array([4.])),.005)
        self.assertEqual(n.observe(np.array([5.]),extrinsic_reward=.1),0)
        self.assertEqual(n.observe(np.array([6.])),0)

if __name__=='__main__':unittest.main()
