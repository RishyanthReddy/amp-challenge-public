import tempfile, unittest
from pathlib import Path
from amp_data.core import csv_rows, fasta_rows, norm, validity, lev_ratio, sequence_id, challenge_matches, xlsx_rows
from amp_data.build import similarity_clusters, clustered_split
from amp_data.core import biophysical_descriptors, molecular_weight_da, boman_index, parse_mic
from amp_data.synthesis_filter import filter_sequences, is_synthesizable
from amp_data.validate import main as validate_main
class CoreTests(unittest.TestCase):
 def test_normalize_and_invalid(self):
  self.assertEqual(norm(' ak k\n'),'AKK'); self.assertFalse(validity('AKBX')['alphabet_valid'])
 def test_length(self): self.assertTrue(validity('ACDEFGHI')['length_valid']); self.assertFalse(validity('ACD')['length_valid'])
 def test_fasta(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.fa';p.write_text('>a test\nACDEFGHI\n>b\nKKKKKKKK\n'); r,_,m=fasta_rows(p);self.assertEqual((len(r),m),(2,0))
 def test_csv_reader_preserves_quoted_newlines(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv';p.write_text('id,note\n1,"first line\nsecond line"\n'); rows,_,_=csv_rows(p);self.assertEqual(rows[0]['note'].splitlines(),['first line','second line'])
 def test_xlsx_reader_preserves_blank_columns(self):
  from zipfile import ZipFile
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.xlsx'
   with ZipFile(p,'w') as z:z.writestr('xl/worksheets/sheet1.xml','''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>first</t></is></c><c r="B1" t="inlineStr"><is><t>middle</t></is></c><c r="C1" t="inlineStr"><is><t>last</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>one</t></is></c><c r="C2" t="inlineStr"><is><t>three</t></is></c></row></sheetData></worksheet>''')
   _,rows,_=xlsx_rows(p)[0];self.assertEqual(rows,[{'first':'one','middle':'','last':'three'}])
 def test_identity(self): self.assertEqual(sequence_id('AAAA'),sequence_id('AAAA'));self.assertGreater(lev_ratio('ACDE','ACDF'),.7)
 def test_official_indel_semantics(self):
  # One substitution costs one deletion + one insertion: 2/(4+4) => 0.75.
  self.assertEqual(lev_ratio('ACDE','ACDF'), .75)
 def test_challenge_overlap(self): self.assertEqual(challenge_matches(['ACDEFGHI'], {'ACDEFGHI'}), [('ACDEFGHI',1.0)])
 def test_clustered_split_is_deterministic_and_group_safe(self):
  rows=[{'sequence_id':'a','sequence':'ACDEFGHI'},{'sequence_id':'b','sequence':'ACDEFGHV'},{'sequence_id':'c','sequence':'KKKKKKKK'}]
  _,_,m=similarity_clusters(rows); one=clustered_split(rows,m[.8]); two=clustered_split(rows,m[.8])
  self.assertEqual([(x['sequence_id'],x['split']) for x in one],[(x['sequence_id'],x['split']) for x in two])
  self.assertEqual({x['split'] for x in one if x['sequence_id'] in {'a','b'}}, {next(x['split'] for x in one if x['sequence_id']=='a')})
 def test_biophysical_descriptors(self):
  d=biophysical_descriptors('AKRKLVW'); self.assertEqual(d['molecular_weight_da'],molecular_weight_da('AKRKLVW')); self.assertGreater(d['net_charge_ph7'],1); self.assertGreater(d['eisenberg_hydrophobic_moment'],0)
 def test_boman_uses_radzicka_wolfenden_scale(self):
  self.assertEqual(boman_index('GLFDIVKKVVGALGSL'), -1.01)
 def test_parse_mic_scientific_censored_and_inverted_range(self):
  sequence='ACDEFGHI'
  self.assertEqual(parse_mic('1.5e-2', 'uM', sequence)['mic_value_numeric'], .015)
  low=parse_mic('<8', 'uM', sequence); self.assertEqual((low['mic_lower_bound'], low['mic_upper_bound']), (None, 8))
  high=parse_mic('>8', 'uM', sequence); self.assertEqual((high['mic_lower_bound'], high['mic_upper_bound']), (8, None))
  interval=parse_mic('16-8', 'uM', sequence); self.assertEqual((interval['mic_lower_bound'], interval['mic_upper_bound']), (8, 16))
  spaced_interval=parse_mic('16 - 8', 'uM', sequence); self.assertEqual((spaced_interval['mic_lower_bound'], spaced_interval['mic_upper_bound']), (8, 16))
 def test_synthesis_filter(self):
  self.assertEqual(is_synthesizable('AKRKLVWQ')[0],True)
  self.assertEqual(is_synthesizable('AAAAAAXX')[1],'invalid_residue')
  self.assertEqual(is_synthesizable('AKRKKLLLLL')[1],'hydrophobic_run')
  self.assertEqual(is_synthesizable('ACDEFGHI')[1],'insufficient_charge')
  self.assertEqual(is_synthesizable('AKRCLVWQ'),(True,'warn_unpaired_cys'))
  batch=filter_sequences(['AKRKLVWQ','AAAAAAXX']); self.assertEqual(batch.to_dict('records')[1]['rejection_reason'],'invalid_residue')
 def test_validator_rejects_empty_and_malformed_fasta(self):
  with tempfile.TemporaryDirectory() as d:
   reference=Path(d)/'reference.fa';reference.write_text('>ref\nACDEFGHI\n')
   for name,contents in [('empty.fa',''),('malformed.fa','ACDEFGHI\n')]:
    candidate=Path(d)/name;candidate.write_text(contents)
    with self.assertRaises(SystemExit): validate_main([str(candidate),'--reference',str(reference)])
if __name__=='__main__':unittest.main()
