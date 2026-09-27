import unittest
import numpy as np
from rware import Warehouse,decode,outcome

class NativeWarehouseTests(unittest.TestCase):
    def test_seeded_native_steps_and_zero_reward_buffers(self):
        a,b=Warehouse(19),Warehouse(19)
        try:
            rng=np.random.default_rng(3)
            for _ in range(256):
                actions=rng.integers(0,5,size=2).tolist();a.step(actions);b.step(actions)
                for i in range(2):
                    np.testing.assert_array_equal(a.observation(i),b.observation(i))
                    self.assertEqual(a.reward(i),b.reward(i))
                    self.assertEqual(a.observation(i).shape,(27,))
                self.assertEqual(a.state(),b.state())
            a.step([0,0]);self.assertEqual([a.reward(0),a.reward(1)],[0.,0.])
        finally:a.close();b.close()
    def test_turn_is_one_native_rotation_without_movement(self):
        e=Warehouse(7)
        try:
            before=e.observation(0);d=decode(before)
            e.step([2,0]);after=e.observation(0)
            self.assertEqual(d['position'],decode(after)['position'])
            self.assertNotEqual(d['direction'],decode(after)['direction'])
            self.assertIn('now facing',outcome(before,after,2,0))
        finally:e.close()

if __name__=='__main__':unittest.main()
