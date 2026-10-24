from pathlib import Path
import os, sys, unittest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'tools'))
from project_paths import ProjectLayout
if not os.environ.get('NADES_DATA_ROOT'):
    raise unittest.SkipTest('Set NADES_DATA_ROOT to the supplied Curated folder.')
layout=ProjectLayout(os.environ['NADES_DATA_ROOT']);D=layout.root
class PathsTests(unittest.TestCase):
    def test_all_312_file_routes(self):
        self.assertEqual(len(layout.rows),312)
        for r in layout.rows:
            self.assertEqual(layout.path(r['original_project_path']),D/r['package_path'])
            self.assertTrue(layout.path(r['original_project_path']).is_file())
    def test_endpoint_collision(self):
        self.assertEqual(layout.path('original.csv'),D/'classification/input/original.csv')
        self.assertEqual(layout.path('TG/original.csv'),D/'regression/input/original.csv')
        self.assertEqual(layout.path('TG/blind2/a.csv'),D/'regression/blind_2/a.csv')
    def test_author_workbook_names(self):
        self.assertEqual(layout.path('Table1-4.xlsx'),D/'classification/input/Table1-3.xlsx')
        self.assertEqual(layout.path('TG/Table1-4.xlsx'),D/'regression/input/Table1-3.xlsx')
    def test_parent_and_solution_matrices(self):
        for name in ['master_train_features1.csv','master_train_features.csv','master_blind_features1.csv','master_blind_features.csv']:
            self.assertEqual(layout.path('TG/'+name),D/'regression/encoded'/name)
    def test_existing_result_directory(self):
        self.assertEqual(layout.path('Final60_Internal_Evaluation'),D/'classification/result/Final60_Internal_Evaluation')
    def test_unseen_plot_output(self):
        self.assertEqual(layout.path('Final60_Blind_Family_Probe/new_plot.pdf'),D/'classification/blind/Final60_Blind_Family_Probe/new_plot.pdf')
    def test_saved_blind_predictions(self):
        self.assertEqual(layout.path('Final60_blind_predictions.csv'),D/'classification/blind/Final60_blind_predictions.csv')
    def test_parent_groups_canonical_copy(self):
        self.assertEqual(layout.path('TG/TG_core_nades_groups.csv'),D/'regression/result/TG_core_nades_groups.csv')
    def test_common_family(self):
        self.assertEqual(layout.path('CrossTask_Family_Multisplit_RF/new.png'),D/'CrossTask_Family_Multisplit_RF/new.png')
    def test_stage_base_dynamic_components(self):
        self.assertEqual(layout.path('')/'a.csv',D/'classification/input/a.csv')
        self.assertEqual(layout.path('TG')/'d.csv',D/'regression/input/d.csv')
    def test_optional_swap_output(self):
        self.assertEqual(layout.path('master_train_features_AB_position_swapped.csv'),D/'classification/encoded/master_train_features_AB_position_swapped.csv')
    def test_no_traversal(self):
        for bad in ['../original.csv','TG/../../a.csv','/etc/passwd','C:\\test.csv']:
            with self.assertRaises(ValueError):layout.path(bad)
    def test_parent_curated_directory(self):
        self.assertEqual(ProjectLayout(D.parent).root,D)

if __name__=='__main__':unittest.main()
