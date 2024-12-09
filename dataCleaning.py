import pandas as pd
from sklearn.impute import KNNImputer

originalData = pd.read_csv('MS-Classification-Research\\De-identified ARR Dataset - new.csv')
#print(originalData.head())

headers = list(originalData.columns.values)
badHeaders = [
    'id', 'BMI', 'Number of Future Relapses 1monthto3years', 
    'Number of Future Relapses 1yrto3yrs', 
    'NfLValue', 'HighNfL Binary'
]
for header in badHeaders:
    headers.remove(header)

initialColumnsDropped = originalData[headers]

blankRowsDropped = initialColumnsDropped.dropna(thresh=26)
blankRowsDropped = blankRowsDropped.dropna(subset=['BOWEL_BLADDER_FUNCTION', 'VISUAL_FUNCTION', 'BRAINSTEM_FUNCTION', 'DiseasedurationatFV'])

blankRowsDropped = blankRowsDropped.loc[blankRowsDropped['VISUAL_FUNCTION'] != 'X']
blankRowsDropped = blankRowsDropped.loc[blankRowsDropped['CEREBELLAR_FUNCTION'] != 'X']
blankRowsDropped = blankRowsDropped.loc[blankRowsDropped['BOWEL_BLADDER_FUNCTION'] != 'X']


blankRowsDropped['TreatmentBeforeFV'] = blankRowsDropped['TreatmentBeforeFV'].replace({'N': 0, 'Y': 1})
blankRowsDropped.to_csv('allAlterations.csv', index=True)
print(blankRowsDropped.shape)
blankRowsDropped = blankRowsDropped.reset_index()

edssKNNImputationCols = ['AgeatFV', 'DiseasedurationatFV', 'PYRAMIDAL_FUNCTION', 
                         'CEREBELLAR_FUNCTION', 'BRAINSTEM_FUNCTION', 'SENSORY_FUNCTION',
                         'BOWEL_BLADDER_FUNCTION', 'VISUAL_FUNCTION', 'MENTAL_FUNCTION',
                         'TotalnumberofrelapsesbeforeFV', 'Numberofrelapsesinthe3yearsbeforeFV',
                         'Numberofrelapsesinthe1yearbeforeFV', 'timeSinceLastAttack',
                         'TreatmentBeforeFV', 'RelapseInYearBeforeFVBinary', 'EDSS_FV',
                         'RelapseInThe3YearsBeforeFVBinary', 'Treatment with Injectable Med']

edssKNNImputationData = blankRowsDropped[edssKNNImputationCols]
print('Imputation Data Initial Shape')
print(edssKNNImputationData.shape)
yeet = edssKNNImputationData.isna().value_counts()
print(yeet)
yeet.to_csv('True-False-Table-Imputation-Data.csv', index=True)
edssKNNImputationData.to_csv('edssKNNImputationData.csv', index=True)


imputer = KNNImputer(n_neighbors=10)
edssImputed = pd.DataFrame(imputer.fit_transform(edssKNNImputationData), columns=edssKNNImputationData.columns)
edssImputed['EDSS_FV'].to_csv('Imputed Data', index=True)
blankRowsDropped['EDSS_FV'].to_csv('Unimputed Data', index=True)
blankRowsDropped['EDSS_FV'] = edssImputed['EDSS_FV']
print(blankRowsDropped.isna().value_counts())
assert blankRowsDropped.index.equals(edssImputed.index)

blankRowsDropped.iloc[:, 1:].to_csv('Fully-Cleaned-Data.csv', index=False)

#pd.DataFrame(edssImputed).to_csv('imputed.csv', index=False)