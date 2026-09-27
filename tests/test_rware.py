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


from rware import ObservationMemory,Prompt

class ObservationMemoryTests(unittest.TestCase):
    @staticmethod
    def obs(x=3,y=3):
        a=np.zeros(27,dtype=np.float32);a[0]=(y*10+x)/110;a[1]=.25
        a[5::3]=.25
        return a
    def test_known_locations_persist_but_requests_update_when_reobserved(self):
        a=self.obs();a[5]=.75;a[11]=1.
        m=ObservationMemory();m.observe(a)
        far=self.obs(8,8);m.tick=15;m.observe(far)
        text=m.describe(far)
        self.assertIn('(3,2) [age 15]',text);self.assertIn('(4,3) [age 15]',text)
        m.observe(self.obs());self.assertNotIn((3,2),{p for p,(tile,t) in m.tiles.items() if tile=='requested shelf'})
        self.assertIn((3,2),m.shelf_sites)
    def test_memory_is_per_agent_and_history_is_bounded(self):
        a=self.obs();a[5]=.75;m=ObservationMemory(8);other=ObservationMemory(8)
        m.observe(a);self.assertEqual(other.tiles,{})
        for _ in range(20):m.record(a,a,0,0)
        self.assertEqual(len(m.history),8);self.assertTrue(m.history[0].startswith('Step 13:'))
    def test_observed_carried_shelf_is_not_a_storage_site(self):
        a=self.obs();a[5]=.75;a[3]=1.;a[4]=.25
        m=ObservationMemory();m.observe(a);self.assertNotIn((3,2),m.shelf_sites)
    def test_prompt_has_coordinates_and_no_unseen_goals(self):
        p=Prompt(dict(history_window=8,rules='Choose a native button.'))
        text=p.build(self.obs(),ObservationMemory())
        self.assertIn('up (3,2): empty floor.',text)
        self.assertIn('Delivery goals: none seen',text)

if __name__=='__main__':unittest.main()
