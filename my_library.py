def test_load():
  return 'MAYA IS STINKY'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def prior_prob(table, target, targetVal):
  columnList = up_get_column(table, target)
  pA = sum([1 if i == targetVal else 0 for i in columnList])/len(columnList)

  return pA

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor

def cond_probs_product(table, evidenceRow, target, targetVal):
  zipColumnRow = up_zip_lists(table.columns[:-1], evidenceRow)
  probList = []

  for i, j in zipColumnRow:
      probList += [cond_prob(table, i, j, target, targetVal)]

  return up_product(probList)

def naive_bayes(table, evidence_row, target):
  cond_prob_no = cond_probs_product(table, evidence_row, target, 0)
  prior_prob_no = prior_prob(table, target, 0)

  cond_prob_yes = cond_probs_product(table, evidence_row, target, 1)
  prior_prob_yes = prior_prob(table, target, 1)

  prob_target_no = cond_prob_no * prior_prob_no
  prob_target_yes = cond_prob_yes * prior_prob_yes
  
  neg, pos = compute_probs(prob_target_no, prob_target_yes)
  return [neg, pos]

def metrics(zipped_list):
  assert isinstance(zipped_list, (list, tuple, set)), 'Paramater is not a list'
  assert all([isinstance(i, (list, tuple, set)) for i in zipped_list]), 'parameter is not list of lists'
  assert all([len(i) == 2 for i in zipped_list]), 'Each value is not a pair'
  assert all([i.__class__ == int and j.__class__ == int for i, j in zipped_list]), 'Each value must be an int'
  assert all([i and j >= 0 for i, j in zipped_list]), 'Each value must be >= 0'

  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  precision = tp / (tp + fp) if (tp + fp) else 0
  recall = tp / (tp + fn) if (tp + fn) else 0
  accuracy = (tp + tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

  return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}
