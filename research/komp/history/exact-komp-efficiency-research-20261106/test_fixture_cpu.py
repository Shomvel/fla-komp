import tempfile,unittest
from pathlib import Path
from fixture_preflight import validate
class FixtureTests(unittest.TestCase):
 def test_missing_dependencies_before_output_creation(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'evidence') as tmp:
   root=Path(tmp)
   with self.assertRaises(FileNotFoundError) as error:validate(root/'checkpoint',root/'manifest.json',root/'audit.json')
   for name in ['manifest.json','audit.json','model.pt','rank0.pt','rank7.pt','complete.json']:self.assertIn(name,str(error.exception))
   self.assertEqual(list(root.iterdir()),[])
if __name__=='__main__':unittest.main()
