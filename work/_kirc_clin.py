# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
c = pd.read_csv('data/raw/TCGA-xena/TCGA-KIRC.clinical.tsv.gz', sep='\t', compression='gzip', low_memory=False)
print('cols of interest:', [x for x in c.columns if any(k in x.lower() for k in ['sample','stage','grade','age'])][:15])
c = c[['sample','ajcc_pathologic_stage.diagnoses','tumor_grade.diagnoses','age_at_diagnosis.diagnoses','age_at_index.demographic']].copy()
c.columns = ['sample_id','stage','grade','age','age2']
c['stage_num'] = c['stage'].str.extract(r'Stage ([IV]+)').replace({'I':1,'II':2,'III':3,'IV':4}).astype(float)
c['grade_num'] = c['grade'].str.extract(r'G([1-4])').astype(float)
print(c[['stage','stage_num','grade','grade_num']].value_counts().head(10))
print('stage non-null:', c['stage_num'].notna().sum(), 'grade non-null:', c['grade_num'].notna().sum())
c.to_csv('data/processed/KIRC/clinical.csv', index=False)
print('saved', len(c), 'rows')
